# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example


class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key, filter_pred=None, corpus_id=None):
        self.sort_key = sort_key
        can_copy = 'src_map' in fields and 'alignment' in fields

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_ in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            """
            ex_dict is one of items in the dataseet with a dict format
            {
             src: 'public void Method_0 ( ) { Varl_0 = false ; try { Varl_1 . Method_1 ( ) ; } catch 
                   ( java.io.IOException Varl_2 ) { Varl_2 . Method_2 ( ) ; error = String_0 + 
                   ( Varl_2 . toString ( ) ) ; } }
             
             tgt: 'public void Method_0 ( ) { Varl_1 . setText ( Varl_2 ) ; Varl_0 . Method_1 ( Varl_1 ) ; 
                   Varl_0 . Method_2 ( new Type_0 ( ) ) ; }'
             'align': '48-12 0-0 47-11 7-7 8-8 45-9 46-10 5-5 2-2 3-3 4-4 1-1\n'
             src_pos: '[108,3,15,22] [108,5,15,22] [108,4,15,22] [108,0,15,22] [108,2,16,23] [108,0,14,25,32]
                       [108,3,10,18,25,37,48,55] [108,0,11,18,30,41,48] [108,4,12,19,31,42,49] [108,6,10,22,33,40]
                       [108,4,15,26,33] [108,0,10,19,30,41,48] [108,3,11,18,26,35,46,57,64] [108,0,11,18,26,35,46,57,64]
                       [108,4,12,19,27,36,47,58,65] [108,0,12,19,27,36,47,58,65] [108,2,13,20,28,37,48,59,66]
                       [108,6,10,18,27,38,49,56] [108,1,11,20,31,42,49] [108,4,12,23,34,41] [108,0,12,20,31,42,49] 
                       [108,4,11,19,29,37,48,59,66] [108,4,11,21,29,40,51,58] [108,1,13,21,32,43,50] 
                       [108,0,12,22,30,41,52,59] [108,4,11,21,29,40,51,58] [108,0,11,18,28,38,46,57,68,75] 
                       [108,4,12,19,29,39,47,58,69,76] [108,0,12,19,29,39,47,58,69,76] [108,2,13,20,30,40,48,59,70,77] 
                       [108,6,10,20,30,38,49,60,67] [108,3,10,18,26,35,45,53,64,75,82] [108,0,11,19,28,38,46,57,68,75] 
                       [108,3,12,20,28,37,47,55,66,77,84] [108,0,12,20,28,37,47,55,66,77,84] [108,0,11,20,28,36,45,55,63,74,85,92] 
                       [108,4,11,21,29,40,51,58] [108,0,11,20,29,37,45,54,64,72,83,94,101] 
                       [108,4,12,21,30,38,46,55,65,73,84,95,102] [108,0,12,21,30,38,46,55,65,73,84,95,102] 
                       [108,2,13,22,31,39,47,56,66,74,85,96,103] [108,1,12,21,29,37,46,56,64,75,86,93] 
                       [108,6,11,20,30,38,49,60,67] [108,1,13,23,31,42,53,60] [108,1,15,26,33]'
             src_path: 'public @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ Modifier | 
                        java.util.List @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType | 
                        < @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType | 
                        TYPE1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType ↑ ClassOrInterfaceType | 
                        > @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType | 
                        METHOD1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration | 
                        ( @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration | 
                        ) @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration | 
                        { @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt | 
                        java.util.ArrayList @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType | 
                        < @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType | 
                        TYPE1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType ↑ ClassOrInterfaceType | 
                        > @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType | 
                        VAR1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator | 
                        = @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator | 
                        new @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr | 
                        java.util.ArrayList @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr ↑ ClassOrInterfaceType | 
                        < @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr ↑ ClassOrInterfaceType | 
                        TYPE1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr ↑ ClassOrInterfaceType ↑ ClassOrInterfaceType | 
                        > @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr ↑ ClassOrInterfaceType | ( @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr | 
                        ) @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ObjectCreationExpr | 
                        ; @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ExpressionStmt | 
                        for @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt | 
                        ( @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt | 
                        TYPE2 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType | 
                        . @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType | 
                        TYPE3 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator ↑ ClassOrInterfaceType | 
                        VAR2 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ VariableDeclarationExpr ↑ VariableDeclarator | 
                        : @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt | 
                        IDENT1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ NameExpr | 
                        ) @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt | 
                        { @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt | 
                        VAR1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr ↑ NameExpr | 
                        . @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr | 
                        METHOD2 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr | 
                        ( @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr | 
                        VAR2 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr ↑ MethodCallExpr ↑ NameExpr | 
                        . @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr ↑ MethodCallExpr | 
                        METHOD1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr ↑ MethodCallExpr | 
                        ( @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr ↑ MethodCallExpr | 
                        ) @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr ↑ MethodCallExpr | 
                        ) @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt ↑ MethodCallExpr | 
                        ; @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt ↑ ExpressionStmt | 
                        } @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ForEachStmt ↑ BlockStmt | return @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ReturnStmt | VAR1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ReturnStmt ↑ NameExpr | ; @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ReturnStmt | } @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt\n'
             tgt_pos: '[74,3,15,22] [74,5,15,22] [74,4,15,22] [74,0,15,22] [74,2,16,23] [74,0,15,26,33] 
                       [74,3,12,21,28,41,52,59] [74,0,12,20,32,43,50] [74,4,13,21,33,44,51] [74,0,13,21,33,44,51] 
                       [74,3,12,21,29,41,52,59] [74,2,14,22,34,45,52] [74,6,11,23,34,41] [74,3,10,19,26,39,50,57] 
                       [74,0,12,19,32,43,50] [74,4,13,20,33,44,51] [74,0,13,20,33,44,51] [74,3,12,21,28,41,52,59] 
                       [74,2,14,21,34,45,52] [74,6,10,23,34,41] [74,3,10,19,26,39,50,57] [74,0,12,21,32,43,50] 
                       [74,4,13,22,33,44,51] [74,0,13,22,33,44,51] [74,1,13,22,31,42,53,60] [74,4,11,22,31,40,51,62,69] 
                       [74,2,14,23,32,43,54,61] [74,8,15,24,33,44,55,62] [74,2,14,23,34,45,52] [74,6,12,23,34,41] 
                       [74,1,16,27,34]' 
             tgt_path: 'public @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ Modifier | 
                        java.util.List @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType | 
                        < @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType | 
                        TYPE1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType ↑ ClassOrInterfaceType | 
                        > @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ ClassOrInterfaceType | 
                        METHOD1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration | 
                        ( @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration | 
                        ) @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration | 
                        { @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt | 
                        return @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ReturnStmt | 
                        VAR1 @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ReturnStmt ↑ NameExpr | 
                        ; @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt ↑ ReturnStmt | 
                        } @ CompilationUnit ↑ ClassOrInterfaceDeclaration ↑ MethodDeclaration ↑ BlockStmt\n'
            } 
            """
            if corpus_id is not None:
                ex_dict["corpus_id"] = corpus_id
            else:
                ex_dict["corpus_id"] = "train"
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(ex_dict, src_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if k in ex_dict}

            # An instance of Example with src, tgt, src_pos, tgt_pos fields. The content of each fields is a list,
            # Rather than a string, of a item.
            ex = Example.fromdict(data=ex_dict, fields=ex_fields)

            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
    def config(fields):
        readers, data, dirs = [], [], []
        for name, field in fields:
            if field["data"] is not None:
                readers.append(field["reader"])
                data.append((name, field["data"]))
                dirs.append(field["dir"])
        return readers, data, dirs
