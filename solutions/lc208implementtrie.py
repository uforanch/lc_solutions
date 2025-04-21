
class Trie:
    def __init__(self):
        self.data = dict()

    def insert(self, word: str) -> None:
        layer =self.data
        word = word +"*"
        for c in word:
            if c not in layer:
                layer[c ] =dict()
                layer = layer[c]
            else:
                layer = layer[c]

    def search(self, word: str) -> bool:
        word = word +"*"
        layer =self.data
        for c in word:
            if c not in layer:
                return False
            else:
                layer = layer[c]
        return True

    def startsWith(self, prefix: str) -> bool:
        layer =self.data
        for c in prefix:
            if c not in layer:
                return False
            else:
                layer = layer[c]
        return True



    # Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


def run_tests(ops, vals):
    outputs = []
    op_dict = {"Trie":Trie}
    for op, val in zip(ops, vals):
        if op == "Trie":
            trie_instance = op(*val)
            op_dict["insert"]=trie_instance.insert
            op_dict["search"]=trie_instance.search
            op_dict["startsWith"]=trie_instance.startsWith
            outputs.append(None)
        else:
            outputs.append(op_dict[op](*val))
    return outputs

#["Trie","insert","search","search","startsWith","insert","search"]
#[[],["apple"],["apple"],["app"],["app"],["app"],["app"]]
#[null,null,true,false,true,null,true]


