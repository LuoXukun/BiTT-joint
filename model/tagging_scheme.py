#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.16

""" Bidirectional Tree Tagging Scheme. """

import copy
import torch

from model.binary_tree import BinaryTree

class BidirectionalTreeTaggingScheme(object):
    """ BiTT Tagging Scheme. """
    def __init__(self, max_seq_len, categories_num):
        super(BidirectionalTreeTaggingScheme, self).__init__()

        self.max_seq_len = max_seq_len
        self.categories_num = categories_num

        self.tag2id = [
            {"NULL": 0, "O": 1, "B": 2, "I": 3, "E": 4, "S": 5},
            {"NULL": 0, "O": 1, "left-1": 2, "left-2": 3, "right-1": 4, "right-2": 5, "root": 6, "right-brother": 7},
            {"NULL": 0, "O": 1, "1": 2, "2": 3, "None": 4},
            {"NULL": 0, "O": 1, "1": 2, "2": 3, "None": 4, "brother": 5}
        ]
        self.id2tag = [
            ["NULL", "O", "B", "I", "E", "S"],
            ["NULL", "O", "left-1", "left-2", "right-1", "right-2", "root", "right-brother"],
            ["NULL", "O", "1", "2", "None"],
            ["NULL", "O", "1", "2", "None", "brother"]
        ]
        self.other_tags, self.begin_tags, self.inner_tags, self.root_tags, self.none_tags = [0, 1], [2, 5], [3, 4], [6, 7], [4]
        self.parts_num = 4
        self.tags_num = [6, 8, 5, 6]
        self.parts_weight = [1.3, 1.0, 1.6, 1.3]
        self.tags_weight = [[] for _ in range(self.parts_num)]
        for i in range(self.parts_num):
            self.tags_weight[i] = [1.0 for j in range(self.tags_num[i])]
            self.tags_weight[i][0] = 0
            self.tags_weight[i][1] = 0.1
    
    def encode_rel_to_bitt_tag(self, sample):
        """ 
            BiTT tags encoder. 
            Args:
                sample:                         {
                                                    "tokens": ["a", "b", "c", ...],
                                                    "relations": [
                                                        {
                                                            "label": "",
                                                            "label_id": number,
                                                            "subject": ["a"],
                                                            "object": ["c"],
                                                            "sub_span": (0, 1),
                                                            "obj_span": (2, 3)
                                                        }, ...
                                                    ]
                                                }
            Returns:
                bitt_tags:                      [tags_forward + tags_backward, ...]
        """
        tokens_len, bitt_tags = len(sample["tokens"]), []

        for cate_id in range(self.categories_num):
            # Get spots list.
            ent_spots, rel_spots = self.__get_spots_list__(sample, cate_id)

            # Forward tags.
            ent_spots_forward, rel_spots_forward = copy.deepcopy(ent_spots), copy.deepcopy(rel_spots)
            self.__sort_spots__(ent_spots_forward, reverse=False)
            root_forward = self.__build_binary_tree_fr_spots__(ent_spots_forward, rel_spots_forward)
            tags_forward = [[self.tag2id[0]["O"] for i in range(tokens_len)] for j in range(self.parts_num)]
            self.__get_tags_fr_binary_tree__(root_forward, tags_forward)

            # Backward tags.
            ent_spots_backward, rel_spots_backward = copy.deepcopy(ent_spots), copy.deepcopy(rel_spots)
            self.__sort_spots__(ent_spots_backward, reverse=True)
            root_backward = self.__build_binary_tree_fr_spots__(ent_spots_backward, rel_spots_backward)
            tags_backward = [[self.tag2id[0]["O"] for i in range(tokens_len)] for j in range(self.parts_num)]
            self.__get_tags_fr_binary_tree__(root_backward, tags_backward)

            # Cut or pad.
            for i in range(self.parts_num):
                if self.max_seq_len > tokens_len:   # Pad.
                    tags_forward[i] += [self.tag2id[i]["NULL"] for _ in range(self.max_seq_len - tokens_len)]
                    tags_backward[i] += [self.tag2id[i]["NULL"] for _ in range(self.max_seq_len - tokens_len)]
                else:                               # Cut.
                    tags_forward[i] = tags_forward[i][:self.max_seq_len]
                    tags_backward[i] = tags_backward[i][:self.max_seq_len]

            # Tensor.
            for i in range(self.parts_num):
                tags_forward[i] = torch.LongTensor(tags_forward[i])
                tags_backward[i] = torch.LongTensor(tags_backward[i])
            
            # bitt_tags.
            bitt_tags.append(torch.stack(tags_forward + tags_backward, 0))
        
        # bitt_tags. (categories_num, parts_num * 2, seq_len)
        bitt_tags = torch.stack(bitt_tags, 0)

        return bitt_tags

    def __get_spots_list__(self, sample, cate_id):
        '''
            Get the spots list.
            Args:
                sample:                     {
                                                "tokens": ["a", "b", "c", ...],
                                                "relations": [
                                                    {
                                                        "label": "",
                                                        "label_id": number,
                                                        "subject": ["a"],
                                                        "object": ["c"],
                                                        "sub_span": (0, 1),
                                                        "obj_span": (2, 3)
                                                    }, ...
                                                ]
                                            }
                cate_id:                    category index.
            Returns:
                ent_spots:                  [(span_pos1, span_pos2)]
                rel_spots:                  [[(sub_span_pos1, sub_span_pos2), (obj_span_pos1, obj_span_pos1), isvalid=True]]
        '''
        ent_spots, rel_spots = set(), []

        for rel in sample["relations"]:
            if rel["label_id"] != cate_id: continue
            sub_tok_span, obj_tok_span = rel["sub_span"], rel["obj_span"]
            if sub_tok_span[1] > self.max_seq_len or obj_tok_span[1] > self.max_seq_len: continue
            ent_spots.add(sub_tok_span)
            ent_spots.add(obj_tok_span)
            rel_spots.append([sub_tok_span, obj_tok_span, True])
        
        return list(ent_spots), rel_spots

    def __sort_spots__(self, entity_spots_list, reverse=False):
        """ 
            Sort the entity_spots_list based on the first element.
            Args:
                entity_spots_list:      The entity spots list.
                reverse:                False -> From small to large. True -> From large to small.
        """
        def take_first(element):
            return element[0]
        
        entity_spots_list.sort(key=take_first, reverse=reverse)
    
    def __find_rel_between_tree_and_node__(self, root, node, rel_spots):
        """ 
            Find if there is a relation between the node in the tree and the new node
            Args:
                root:           The Binary Tree root
                node:           The new node
                rel_spots:      The relation spots list
            Return:
                parent:         The parent node of the new node if relation exists
                child_relation: The label of parent node if relation exists
        """
        parent, child_relation = None, ""
        if root:
            for rel in rel_spots:
                if rel[2] == False: continue
                if root.location == rel[0] and node.location == rel[1]:
                    child_relation = "1"        # The node is obj.
                    parent = root
                elif root.location == rel[1] and node.location == rel[0]:
                    child_relation = "2"        # The node is sub.
                    parent = root
            if parent == None and child_relation == "":
                parent, child_relation = self.__find_rel_between_tree_and_node__(root.firstChild, node, rel_spots)
            if parent == None and child_relation == "":
                parent, child_relation = self.__find_rel_between_tree_and_node__(root.nextSibling, node, rel_spots)
        return parent, child_relation
    
    def __build_binary_tree_fr_spots__(self, ent_spots, rel_spots):
        """ 
            Build binary tree from sorted entity spots and rel spots.
            Args:
                ent_spots:                  [(span_pos1, span_pos2)]
                rel_spots:                  [[(sub_span_pos1, sub_span_pos2), (obj_span_pos1, obj_span_pos1), isValid]]
            Returns:
                root:                       The root node of the relation binary tree.
        """
        root = None

        while len(ent_spots) != 0:
            # Get the ahead entity in the sentence, create a tree node from it.
            # Since the ent_spots has been sorted, we simply get the first one.
            node = BinaryTree(ent_spots[0])
            ent_spots.remove(ent_spots[0])

            # Find all children of the node.
            # Strictly promise the location sort of sibling nodes.
            remove_ents, relation_flag = [], False
            for i in range(len(ent_spots)):
                child_pair = ent_spots[i]
                for rel in rel_spots:
                    if rel[2] == False: continue        # If valid.
                    if node.location == rel[0] and child_pair == rel[1]:
                        child_node = BinaryTree(child_pair)
                        child_relation = "1"            # Means that child is obj.
                        node.insertChild(child_node, child_relation)
                        rel[2], relation_flag = False, True
                        remove_ents.append(child_pair)  # The entity to be removed.
                        break                           # Avoid two relations which have the same sub and the same obj
                    elif node.location == rel[1] and child_pair == rel[0]:
                        child_node = BinaryTree(child_pair)
                        child_relation = "2"            # Means that child is sub.
                        node.insertChild(child_node, child_relation)
                        rel[2], relation_flag = False, True
                        remove_ents.append(child_pair)  # The entity to be removed.
                        break
            
            # Insert the node into Binary tree
            if not root:
                root = node
            else:
                parent_node, child_relation = self.__find_rel_between_tree_and_node__(root, node, rel_spots)
                if parent_node != None and child_relation != "":
                    parent_node.insertChild(node, child_relation)
                else:
                    if relation_flag:
                        root.insertSibling(node, "brother")

            # Remove the entities added to the binary tree.
            for item in remove_ents:
                ent_spots.remove(item)
    
        return root
        
    def __get_tags_fr_binary_tree__(self, root, tags):
        """ 
            Tag the sentence according to the relation tree.
            Args:
                root:           The Binary Tree root.
                tags:           The tag list. [4, seq_len]
            Returns:
        """
        if root != None:
            pair = root.location
            changed = False
            for i in range(pair[0], pair[1]):
                if tags[0][i] != self.tag2id[0]["O"]: changed = True
            if changed == False:
                childRelation, siblingRelation = "None", "None"

                if root.childRelation != "": 
                    childRelation = root.childRelation
                if root.siblingRelation != "":
                    siblingRelation = root.siblingRelation

                # BIES+child-lable+parent-left-label+parent-right-label
                if pair[1] - pair[0] == 1:
                    tags[0][pair[0]] = self.tag2id[0]["S"]
                else:
                    tags[0][pair[0]] = self.tag2id[0]["B"]
                    tags[0][pair[1]-1] = self.tag2id[0]["E"]
                    tags[0][pair[0]+1:pair[1]-1] = [self.tag2id[0]["I"] for _ in range(pair[1] - pair[0] - 2)]
                tags[1][pair[0]:pair[1]] = [self.tag2id[1][root.parent] for _ in range(pair[1] - pair[0])]
                tags[2][pair[0]:pair[1]] = [self.tag2id[2][childRelation] for _ in range(pair[1] - pair[0])]
                tags[3][pair[0]:pair[1]] = [self.tag2id[3][siblingRelation] for _ in range(pair[1] - pair[0])]
            
            self.__get_tags_fr_binary_tree__(root.firstChild, tags)
            self.__get_tags_fr_binary_tree__(root.nextSibling, tags)
    
    def decode_rel_fr_bitt_tag(self, tokens, tags):
        """ 
            BiTT tags decoder.
            Args:
                tokens:         ["a", "b", "c", ...]
                tags:           (parts_num * 2, categories_num, tags_len)
            Returns:
                rel_list:       [{
                                    "subject": subject tokens,
                                    "object": object tokens,
                                    "sub_span": subject location pair,
                                    "obj_span": object location pair
                                }, ...]
        """
        rel_list = []
        
        for cate_id in range(self.categories_num):
            # Build the relation tree.
            root_forward = self.__build_relation_tree__(tokens, tags[:self.parts_num, cate_id, :], forward=True)
            root_backward = self.__build_relation_tree__(tokens, tags[self.parts_num:, cate_id, :], forward=False)

            # Get pred relation set.
            relations_forward, relations_backward = set(), set()
            self.__get_relation_fr_tree__(root_forward, relations_forward)
            self.__get_relation_fr_tree__(root_backward, relations_backward)
            relations = relations_forward.union(relations_backward)

            # Get rel_list.
            for relation in list(relations):
                spans = relation.split("-")
                rel_list.append({
                    "subject": tokens[int(spans[0]):int(spans[1])], "object": tokens[int(spans[2]):int(spans[3])],
                    "sub_span": (int(spans[0]), int(spans[1])), "obj_span": (int(spans[2]), int(spans[3])),
                    "label_id": cate_id
                })
        
        return rel_list
    
    def __build_relation_tree__(self, tokens, tags, forward):
        """ 
            Build the relation tree according to the tag and tokens. (Forward.)
            Args:
                text:           ["a", "b", "c", ...]
                tags:           (4, tags_len)
                forward:        True -> Forward, False -> Backward.
            Return:
                root:           the Binary tree
        """
        tokens_len, tags_len = len(tokens), tags[0].size(0)
        seq_len = min(tokens_len, tags_len)
        visit_flag = [False for _ in range(seq_len)]

        # Find all root in the forest.
        root_indexes = []
        if forward: # From right to left.
            cursor = seq_len - 1
            while cursor >= 0:
                if tags[1][cursor] in self.root_tags and tags[0][cursor] in self.begin_tags:
                    for i in range(cursor + 1, seq_len):
                        if tags[0][i] in self.inner_tags: continue
                        root_indexes.append((cursor, i))
                        break
                cursor -= 1
        else:       # From left to right.
            cursor = 0
            while cursor < seq_len:
                if tags[1][cursor] in self.root_tags and tags[0][cursor] in self.begin_tags:
                    for i in range(cursor + 1, seq_len):
                        if tags[0][i] in self.inner_tags: continue
                        root_indexes.append((cursor, i))
                        break
                cursor += 1
        
        # Build the tree.
        # Take the situation that two same entities are in a sentence into account.
        root, brother = None, None
        for root_index in root_indexes:
            root = BinaryTree(root_index)
            self.recurrent = 0
            self.__build_child_tree__(root, tags, visit_flag, seq_len, forward, True)
            if brother:
                root.insertSibling(brother, "brother")
            brother = root
        
        return root
    
    def __build_child_tree__(self, root, tags, visit_flag, seq_len, forward, isRoot):
        """ 
            Build a child tree in the forest.
            Args:
                root:           The root of the child tree.
                tags:           (4, tags_len)
                visit_flag:     (tags_len)
                seq_len:        The seq length.
                forward:        True -> Forward, False -> Backward.
                isRoot:         If this node a root node in the forest.
            Return:
        """
        if not visit_flag[root.location[0]]:
            visit_flag[root.location[0]] = True
        else:
            print("Waring: there is something wrong in building child tree!")
            return
        
        self.recurrent += 1
        if self.recurrent >= 1000:
            print("Warning: Died recurrent!")
            return
        
        # Left Tree, if there exists child node.
        if tags[2][root.location[0]] not in (self.none_tags + self.other_tags):
            parent_label = self.id2tag[2][tags[2][root.location[0]]].split("-")
            assert len(parent_label) == 1
            isFound = False
            if forward:     # Find the node on the right.
                for i in range(root.location[1], seq_len):
                    # Find the begin label and nonvisited label.
                    if tags[0][i] in self.begin_tags and visit_flag[i] == False:
                        # Find the label linked to the parent label.
                        child_label = self.id2tag[1][tags[1][i]].split("-")
                        if len(child_label) != 2: continue
                        if child_label[0] == "left":
                            if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node.
                                isFound = True
                                for j in range(i + 1, seq_len + 1):
                                    if j != seq_len:
                                        if tags[0][j] in self.inner_tags:
                                            continue
                                    child_node = BinaryTree((i, j))
                                    root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                    assert root.firstChild.parent == self.id2tag[1][tags[1][i]]
                                    break
                                if root.firstChild == None:
                                    print("Warning: Cannot find the child node while finding right child node!")
                                    break
                                self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                break
                # If we didn't find child node and current node is not root, we find child node on the left
                if isFound == False and isRoot == False:
                    #print("left2")
                    cursor = root.location[0] - 1
                    #print("cursor: ", cursor)
                    while cursor >= 0:
                        # Find the begin label and nonvisited label
                        if tags[0][cursor] in self.begin_tags and visit_flag[cursor] == False:
                            # Find the label linked to the parent label
                            child_label = self.id2tag[1][tags[1][cursor]].split("-")
                            if len(child_label) != 2: 
                                cursor -= 1
                                continue
                            if child_label[0] == "left":
                                if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                    # We find the child node
                                    #print("We find the child node")
                                    isFound = True
                                    for j in range(cursor + 1, root.location[0] + 1):
                                        if j!= root.location[0]:
                                            if tags[0][j] in self.inner_tags: continue
                                        child_node = BinaryTree((cursor, j))
                                        root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                        assert root.firstChild.parent == self.id2tag[1][tags[1][cursor]]
                                        break
                                    if root.firstChild == None:
                                        print("Warning: Cannot find the child node while finding left child node!")
                                        break
                                    #print("Start 2!", self.recurrent)
                                    self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                    #print("End 2!", self.recurrent)
                                    break
                        cursor -= 1
            else:           # Find the node on the left.
                cursor = root.location[0] - 1
                #print("cursor: ", cursor)
                while cursor >= 0:
                    # Find the begin label and nonvisited label
                    if tags[0][cursor] in self.begin_tags and visit_flag[cursor] == False:
                        # Find the label linked to the parent label
                        child_label = self.id2tag[1][tags[1][cursor]].split("-")
                        if len(child_label) != 2: 
                            cursor -= 1
                            continue
                        if child_label[0] == "left":
                            if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node
                                #print("We find the child node")
                                isFound = True
                                for j in range(cursor + 1, root.location[0] + 1):
                                    if j!= root.location[0]:
                                        if tags[0][j] in self.inner_tags: continue
                                    child_node = BinaryTree((cursor, j))
                                    root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                    assert root.firstChild.parent == self.id2tag[1][tags[1][cursor]]
                                    break
                                if root.firstChild == None:
                                    print("Warning: Cannot find the child node while finding left child node!")
                                    break
                                #print("Start 2!", self.recurrent)
                                self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                #print("End 2!", self.recurrent)
                                break
                    cursor -= 1
                # If we didn't find child node and current node is not root, we find child node on the right
                if isFound == False and isRoot == False:
                    #print("left2")
                    for i in range(root.location[1], seq_len):
                        # Find the begin label and nonvisited label
                        if tags[0][i] in self.begin_tags and visit_flag[i] == False:
                            # Find the label linked to the parent label
                            child_label = self.id2tag[1][tags[1][i]].split("-")
                            if len(child_label) != 2: continue
                            if child_label[0] == "left":
                                if abs(int(child_label[1]) - int(parent_label[0])) == 1:
                                    # We find the child node
                                    isFound = True
                                    for j in range(i + 1, seq_len + 1):
                                        if j != seq_len:
                                            if tags[0][j] in self.inner_tags: continue
                                        child_node = BinaryTree((i, j))
                                        root.insertChild(child_node, self.id2tag[2][tags[2][root.location[0]]])
                                        assert root.firstChild.parent == self.id2tag[1][tags[1][i]]
                                        break
                                    if root.firstChild == None:
                                        print("Warning: Cannot find the child node while finding right child node!")
                                        break
                                    self.__build_child_tree__(root.firstChild, tags, visit_flag, seq_len, forward, False)
                                    break
        
        # Right Tree, if there exists sibling node.
        if tags[3][root.location[0]] not in (self.none_tags + self.other_tags) and tags[3][root.location[0]] != 5:
            parent_label = self.id2tag[3][tags[3][root.location[0]]].split("-")
            #print("parent_label: ", parent_label)
            assert len(parent_label) == 1
            isFound = False
            if forward:     # Find the node on the right.
                for i in range(root.location[1], seq_len):
                    # Find the begin label and nonvisited label
                    if tags[0][i] in self.begin_tags and visit_flag[i] == False:
                        # Find the label linked to the parent label
                        sibling_label = self.id2tag[1][tags[1][i]].split("-")
                        if len(sibling_label) != 2: continue
                        if sibling_label[0] == "right" and sibling_label[1] != "brother":
                            #print("sibling_label: ", sibling_label[1])
                            #print("parent_label: ", parent_label[0])
                            if abs(int(sibling_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node
                                isFound = True
                                for j in range(i + 1, seq_len + 1):
                                    if j != seq_len:
                                        if tags[0][j] in self.inner_tags: continue
                                    sibling_node = BinaryTree((i, j))
                                    root.insertSibling(sibling_node, self.id2tag[3][tags[3][root.location[0]]])
                                    assert root.nextSibling.parent == self.id2tag[1][tags[1][i]]
                                    break
                                if root.nextSibling == None:
                                    print("Warning: Cannot find the sibling node while finding right sibling node!")
                                    break
                                self.__build_child_tree__(root.nextSibling, tags, visit_flag, seq_len, forward, False)
                                break
            else:           # Find the node on the left.
                cursor = root.location[0] - 1
                while cursor >= 0:
                    # Find the begin label and nonvisited label
                    if tags[0][cursor] in self.begin_tags and visit_flag[cursor] == False:
                        # Find the label linked to the parent label
                        sibling_label = self.id2tag[1][tags[1][cursor]].split("-")
                        if len(sibling_label) != 2: 
                            cursor -= 1
                            continue
                        if sibling_label[0] == "right" and sibling_label[1] != "brother":
                            if abs(int(sibling_label[1]) - int(parent_label[0])) == 1:
                                # We find the child node
                                isFound = True
                                for j in range(cursor + 1, root.location[0] + 1):
                                    if j != root.location[0]:
                                        if tags[0][j] in self.inner_tags: continue
                                    sibling_node = BinaryTree((cursor, j))
                                    root.insertSibling(sibling_node, self.id2tag[3][tags[3][root.location[0]]])
                                    assert root.nextSibling.parent == self.id2tag[1][tags[1][cursor]]
                                    break
                                if root.nextSibling == None:
                                    print("Warning: Cannot find the sibling node while finding right sibling node!")
                                    break
                                self.__build_child_tree__(root.nextSibling, tags, visit_flag, seq_len, forward, False)
                                break
                    cursor -= 1
        
        return
    
    def __get_relation_fr_tree__(self, root, relation_set):
        """ 
            Get all predicted relations according the tree we had built.
            Args:
                root:           The root of the Binary tree
                relation_set:   All relations (sub_span_1-sub_span_2-obj_span_1-obj_span2)
            Returns:
        """
        if not root: return
        child_list = root.getChildList()
        if len(child_list):
            for child in child_list:
                label_list = child["label"].split("-")
                assert len(label_list) == 1
                if label_list[0] == "1":
                    relation_set.add("{}-{}-{}-{}".format(root.location[0], root.location[1], child["location"][0], child["location"][1]))
                else:
                    relation_set.add("{}-{}-{}-{}".format(child["location"][0], child["location"][1], root.location[0], root.location[1]))
        self.__get_relation_fr_tree__(root.firstChild, relation_set)
        self.__get_relation_fr_tree__(root.nextSibling, relation_set)
        return