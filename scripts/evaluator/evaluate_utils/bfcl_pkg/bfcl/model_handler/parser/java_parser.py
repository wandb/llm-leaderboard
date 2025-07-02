from tree_sitter import Language, Parser
import tree_sitter_java
import os

# Build and load the Java language
JAVA_LANGUAGE = Language(tree_sitter_java.language(), "java")

parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def parse_java_function_call(source_code):
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node

    if "ERROR" in root_node.sexp():
        raise Exception("Error parsing Java source code")

    def get_text(node):
        """Returns the text represented by the node."""
        return source_code[node.start_byte : node.end_byte]

    def get_arguments(args_node):
        args = {}
        for child in args_node.children:
            if child.type not in [",", "(", ")"]:
                args[None] = get_text(child)
        return args

    def find_method_call(node):
        if node.type == "method_invocation":
            method_name = get_text(node.child_by_field_name("name"))
            class_node = node.child_by_field_name("object")
            
            if class_node:
                method_name = f"{get_text(class_node)}.{method_name}"
            
            args_node = node.child_by_field_name("arguments")
            args = get_arguments(args_node) if args_node else {}
            
            return [{method_name: args}]
        
        for child in node.children:
            result = find_method_call(child)
            if result:
                return result
        
        return None

    result = find_method_call(root_node)
    return result if result else {}
