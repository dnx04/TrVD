import tree_sitter

class ASTNode(object):
    def __init__(self, node, do_split=True):
        self.node = node
        self.do_split = do_split
        self.is_leaf = self.is_leaf_node()
        self.token = self.get_token()
        self.children = self.add_children()

    def is_leaf_node(self):
        # return len(self.node.children) == 0
        if not isinstance(self.node, tree_sitter.Tree):
            return len(self.node.children) == 0
        else:
            return len(self.node.root_node.children) == 0

    def get_token(self) -> str | bytes:
        if not isinstance(self.node, tree_sitter.Tree):
            token = self.node.type
            if self.is_leaf:
                token = self.node.text
            return token  # type: ignore[return-value]
        else:
            token = self.node.root_node.type
            if self.is_leaf:
                token = self.node.root_node.text
            return token  # type: ignore[return-value]

    def add_children(self):
        if self.is_leaf:
            return []
        children = self.node.children
        if not self.do_split:
            return [ASTNode(child, self.do_split) for child in children]
        else:
            if self.token in ['function_definition', 'if_statement', 'try_statement', 'for_statement',
                              'switch_statement',
                              'while_statement', 'do_statement', 'catch_clause', 'case_statement']:
                # find first compound_statement
                body_idx = 0
                for child in children:
                    if child.type == 'compound_statement':
                        break
                    body_idx += 1
                return [ASTNode(children[c], self.do_split) for c in range(0, body_idx)]
            else:
                return [ASTNode(child, self.do_split) for child in children]


class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.is_leaf = self.is_leaf_node()
        self.token = self.get_token()
        self.children = []

    def is_leaf_node(self):
        # return len(self.node.children) == 0
        if not isinstance(self.node, tree_sitter.Tree):
            return len(self.node.children) == 0
        else:
            return len(self.node.root_node.children) == 0

    def get_token(self) -> str | bytes:
        if not isinstance(self.node, tree_sitter.Tree):
            token = self.node.type
            if self.is_leaf:
                token = self.node.text
            return token  # type: ignore[return-value]
        else:
            token = self.node.root_node.type
            if self.is_leaf:
                token = self.node.root_node.text
            return token  # type: ignore[return-value]
