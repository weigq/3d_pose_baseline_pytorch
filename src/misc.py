#!/usr/bin/env python
# -*- coding: utf-8 -*-


def define_actions(action):
    """
    :param action: specified action
    :return: a list of action(s)
    """
    actions = ["Directions",
               "Discussion",
               "Eating",
               "Greeting",
               "Phoning",
               "Photo",
               "Posing",
               "Purchases",
               "Sitting",
               "SittingDown",
               "Smoking",
               "Waiting",
               "WalkDog",
               "Walking",
               "WalkTogether"]

    if action == "All" or action == "all":
        return actions

    if action not in actions:
        raise (ValueError, "Unincluded action: {}".format(action))

    return [action]
