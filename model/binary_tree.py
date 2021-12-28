#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.16

""" Binary Tree class. """

class BinaryTree(object):
    """ 
        Init a Binary Tree to represent a forest

        Args:

            text:               the text of the entity node
            location:           the location in the sentence of the entity

        Members:

            firstChild:         the first child of the node in the forest
            nextSibling:        the next brother of the node in the forest
            childRelation:      the relation between this node and its child in the forest(for parent)
            siblingRelation:    the relation between this node's parent and its brother in the forest(for parent)
            location:           the location in the sentence of the entity
            parent:             the left child or the right child of parent, and the relation(for child)
    """
    def __init__(self, location):
        super(BinaryTree, self).__init__()
        self.location = location
        self.firstChild = None
        self.nextSibling = None
        self.childRelation = ""
        self.siblingRelation = ""
        self.parent = "root"
    
    def insertChild(self, child_node, child_relation):
        """ 
            Insert a child node of this node in the forest
            Args:
                child_node:         A Binary Tree Node, this node's child
                child_relation:     The parent's label from the relation between child and its parent
            Return:
        """
        if child_relation == str(1): label = str(2)
        elif child_relation == str(2): label = str(1)
        else: label = child_relation

        """ if len(label) == 2:
            if label[1] == str(1): label[1] = str(2)
            elif label[1] == str(2): label[1] = str(1)
            label = label[0] + "-" + label[1]
        else:
            label = child_relation """
        if self.firstChild == None:
            child_node.parent = "left-"+ label
            self.firstChild = child_node
            self.childRelation = child_relation
        else:
            self.firstChild.insertSibling(child_node, child_relation)
    
    def insertSibling(self, sibling_node, sibling_relation):
        """ 
            Insert the next sibling of this node in the forest
            Args:
                sibling_node:       A Binary Tree Node, this node's brother
                sibling_relation:   The sibling's label from the relation between sibling and its parent in the forest
            Return:
        """
        if sibling_relation == str(1): label = str(2)
        elif sibling_relation == str(2): label = str(1)
        else: label = sibling_relation
        
        sibling_node.parent = "right-" + label
        if self.nextSibling == None:
            self.nextSibling = sibling_node
            self.siblingRelation = sibling_relation
        else:
            brother = self.nextSibling
            while brother.nextSibling != None:
                brother = brother.nextSibling
            brother.nextSibling = sibling_node
            brother.siblingRelation = sibling_relation

    def getChildList(self):
        """ 
            Get the child list of this node in the forest
            Return:
                child_list:         The child list
        """
        child_list = []
        if self.firstChild != None:
            child = {}
            child["location"] = self.firstChild.location
            child["label"] = self.childRelation
            child["parent"] = self.firstChild.parent
            child_list.append(child)
            
            left, right = self.firstChild, self.firstChild.nextSibling
            while right != None:
                child = {}
                child["location"] = right.location
                child["label"] = left.siblingRelation
                child["parent"] = right.parent
                child_list.append(child)
                left = right
                right = right.nextSibling        
        return child_list

    def get_tree_dict(self, tokens):
        """ 
            Get the information dictionary of the tree.
            Return:
                tree_dict:          The tree dictionary
        """
        tree_dict = {}
        tree_dict["text"] = " ".join(tokens[self.location[0]:self.location[1]])
        tree_dict["location"] = self.location
        if self.firstChild != None:
            tree_dict["firstChild"] = self.firstChild.get_tree_dict()
            tree_dict["childRelation"] = self.childRelation
        if self.nextSibling != None:
            tree_dict["nextSibling"] = self.nextSibling.get_tree_dict()
            tree_dict["siblingRelation"] = self.siblingRelation
        
        return tree_dict
