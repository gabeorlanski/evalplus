"""Post-processing LLM-generated Python code implemented using tree-sitter."""

import os
import pathlib
from typing import Dict, Generator, List, Optional, Set, Tuple

import tree_sitter_python
from tqdm import tqdm
from tree_sitter import Language, Node, Parser

from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
    load_solutions,
    write_directory,
    write_jsonl,
)
from evalplus.syncheck import syntax_check

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


# Mapping node types to categories for spacing rules
type_map = {
    **{t: "import" for t in IMPORT_TYPE},
    FUNCTION_TYPE: "definition",
    CLASS_TYPE: "definition",
    # Consider top-level assignments captured via ExpressionStatement as definitions for spacing
    EXPRESSION_TYPE: "definition",
}


def get_definition_name(node: Node) -> Optional[str]:
    # Handle direct Function/Class definitions
    if node.type in [FUNCTION_TYPE, CLASS_TYPE]:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                return child.text.decode("utf8")
    # Handle assignments (e.g., x = ...) potentially wrapped in ExpressionStatement
    elif (
        node.type == EXPRESSION_TYPE
        and node.children
        and node.children[0].type == ASSIGNMENT_TYPE
    ):
        assign_node = node.children[0]
        # Traverse the assignment node to find the identifier being assigned to
        target_node = assign_node.child_by_field_name("left")
        if target_node and target_node.type == IDENTIFIER_TYPE:
            return target_node.text.decode("utf8")
    elif (
        node.type == ASSIGNMENT_TYPE
    ):  # Handle raw assignment if it's somehow a direct child
        target_node = node.child_by_field_name("left")
        if target_node and target_node.type == IDENTIFIER_TYPE:
            return target_node.text.decode("utf8")

    return None  # Return None if name cannot be extracted


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    if not code:  # Handle empty code after extraction
        return ""

    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)

    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    # Stores tuples of (name, node_object, node_category)
    definition_node_info = []

    for child in root_node.children:
        node_type_str = child.type
        category = type_map.get(node_type_str, "other")
        name = get_definition_name(child)

        if category == "import":
            import_nodes.append(child)
        elif category == "definition" and name:
            # Check for duplicates based on name
            if name in class_names or name in variable_names or name in function_names:
                continue

            # Specific handling based on definition type
            if node_type_str == CLASS_TYPE:
                definition_node_info.append((name, child, category))
                class_names.add(name)
            elif node_type_str == FUNCTION_TYPE:
                # Ensure function has a return statement before adding
                if has_return_statement(child):
                    definition_node_info.append((name, child, category))
                    function_names.add(name)
            elif (
                node_type_str == EXPRESSION_TYPE
                and child.children
                and child.children[0].type == ASSIGNMENT_TYPE
            ):
                # Successfully extracted name from assignment via get_definition_name
                definition_node_info.append((name, child, category))
                variable_names.add(name)

    reacheable = set()
    if entrypoint:
        # Extract just (name, node) pairs for dependency analysis
        definition_nodes_for_deps = [
            (name, node) for name, node, cat in definition_node_info
        ]
        name2deps = get_deps(definition_nodes_for_deps)
        reacheable = get_function_dependency(entrypoint, name2deps)
        # Add the entrypoint itself to reachable if it exists
        all_defined_names = class_names | function_names | variable_names
        if entrypoint in all_defined_names:
            reacheable.add(entrypoint)

    # --- Start Building Output ---
    nodes_to_keep_info = []  # Stores tuples of (node, category)
    kept_node_starts = set()  # Track start bytes to avoid adding duplicates

    # Add necessary imports (Refinement: Ideally, filter based on actual usage by reachable code)
    for node in import_nodes:
        if node.start_byte not in kept_node_starts:
            nodes_to_keep_info.append((node, "import"))
            kept_node_starts.add(node.start_byte)

    # Add reachable definitions
    for name, node, category in definition_node_info:
        if node.start_byte not in kept_node_starts:
            # Keep if no entrypoint OR if name is reachable
            if not entrypoint or name in reacheable:
                nodes_to_keep_info.append((node, category))
                kept_node_starts.add(node.start_byte)

    # Sort all nodes to keep by their starting position in the original code
    nodes_to_keep_info.sort(key=lambda item: item[0].start_byte)

    sanitized_output = b""
    for i, (current_node, current_category) in enumerate(nodes_to_keep_info):
        # Append the code corresponding to the current node
        start = current_node.start_byte
        end = current_node.end_byte
        # Ensure start/end are valid indices for slicing
        if start < end and end <= len(code_bytes):
            sanitized_output += code_bytes[start:end]

        # Determine and append spacing based on the next node, if it exists
        if i < len(nodes_to_keep_info) - 1:
            next_node, next_category = nodes_to_keep_info[i + 1]

            # Apply PEP8-like spacing rules
            if current_category == "import" and next_category == "import":
                sanitized_output += b"\n"
            elif current_category == "import" and next_category == "definition":
                sanitized_output += b"\n\n"
            elif current_category == "definition" and next_category == "definition":
                sanitized_output += b"\n\n"
            elif current_category == "definition" and next_category == "import":
                # Should not happen if imports are first, handle defensively
                sanitized_output += b"\n\n"
            else:  # Default fallback
                sanitized_output += b"\n"

    # Decode the result from bytes to string
    final_code = sanitized_output.decode("utf8")

    return final_code


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    # Pass entrypoint to the extraction function
    sanitized_code = extract_target_code_or_empty(code, entrypoint)
    # Fallback to basic syntax extraction if sanitization yields empty result
    if not sanitized_code.strip():
        return code_extract(code)
    return sanitized_code


def script(
    samples: str, inplace: bool = False, debug_task: str = None, mbpp_version="default"
):
    # task_id -> entry_point
    entry_point = {}
    # merge two datasets
    dataset = {**get_human_eval_plus(), **get_mbpp_plus(version=mbpp_version)}

    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(samples)
    target_path = pathlib.Path(samples)
    if not inplace:
        if is_folder:
            new_name = target_path.name + "-sanitized"
        else:
            new_name = target_path.name.replace(".jsonl", "-sanitized.jsonl")
        target_path = target_path.parent / new_name
    target_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    for solution in tqdm(load_solutions(samples)):
        task_id = solution["task_id"]
        if task_id not in dataset:
            print(
                f"Skiping {task_id} as it does not existing in the latest EvalPlus dataset."
            )
            continue

        function_name = entry_point[task_id] if task_id in entry_point else None
        dbg_identifier = solution["_identifier"]
        if debug_task is not None and task_id != debug_task:
            continue

        ntotal += 1
        if "solution" in solution:
            old_code = solution["solution"]
        else:
            assert "completion" in solution
            old_code = dataset[task_id]["prompt"] + "\n" + solution["completion"]

        new_code = sanitize(code=old_code, entrypoint=function_name)

        # if changed, print the message
        if new_code != old_code:
            msg = "Sanitized: " + dbg_identifier
            if is_folder:
                msg += " -> " + dbg_identifier.replace(samples, target_path)
            print(msg)
            nsan += 1

        new_solutions.append({"task_id": task_id, "solution": new_code})

    if is_folder:
        write_directory(target_path, new_solutions)
    else:
        write_jsonl(target_path, new_solutions)

    if nsan > 0:
        print(f"Sanitized {nsan} out of {ntotal} files.")
    else:
        print(f"All files seems valid -- no files are sanitized.")
    print(f"Check the sanitized files at {target_path}")


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
