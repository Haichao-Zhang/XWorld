from xworld3d_env import XWorld3DEnv
from py_gflags import get_flag
import os
import random
from py_util import overrides
from py_util import tsum

class XWorld3DDiaNavMap(XWorld3DEnv):
    def __init__(self, asset_path, start_level=0):
        super(XWorld3DDiaNavMap, self).__init__(
            asset_path=asset_path,
            max_height=5,
            max_width=5,
            maze_generation=False)
        self.class_per_session = 2 # max number of classes in a session
                                   # value < 1 denotes all classes are used
        self.sel_classes = {} # selected classes for a session
        self.shuffle = False # shuffle classes
        # (y, x, z)
        self.nav_loc_set = [(2, 1, 0), (2, 3, 0)]
        self.dia_loc_set = [(0, 0, 0), (0, 4, 0)]
        self.teach_loc = (0, 2, 0)
        self.agent_yaw_set = [3.14] # [0, 3.14] # yaw set for agent

    def _configure(self, select_class=True):
        self.set_goal_subtrees([ "others", "furniture"])
        ## these are all the object classes
        self.set_dims(5, 5)

        if select_class:
            self.select_goal_classes() # re-select goal class for a new session

        if self.shuffle:
            self.shuffle_classes("goal")

        self.set_entity(type="agent", loc=(2, 2, 0))
        self.set_entity(type="goal", loc=self.nav_loc_set[0])
        self.set_entity(type="goal", loc=self.nav_loc_set[1])
        self.set_entity(type="goal", loc=self.dia_loc_set[0])
        self.set_entity(type="goal", loc=self.dia_loc_set[1])

        sel_goals = self.get_selected_goal_classes()

        # sel_goals.pop()
        # sel_goals += ["carpet"]

        random.shuffle(sel_goals)
        for i, e in enumerate(self.get_nav_goals()):
            self.set_property(e, property_value_dict={"name" : sel_goals[i], \
                                                      "yaw" : 2})
        random.shuffle(sel_goals)
        for i, e in enumerate(self.get_dia_goals()):
            self.set_property(e, property_value_dict={"name" : sel_goals[i], \
                                                      "yaw" : 0})
        a, _, _ = self.get_agent()

        self.agent_yaw = random.choice(self.agent_yaw_set)
        self.set_property(a, property_value_dict={"yaw" : self.agent_yaw})

    def get_nav_goals(self):
        goals = self.get_goals()
        nav_goals = [g for g in goals if g.loc in self.nav_loc_set]
        return nav_goals

    def get_dia_goals(self):
        goals = self.get_goals()
        dia_goals = [g for g in goals if g.loc in self.dia_loc_set]
        return dia_goals

    @overrides(XWorld3DEnv)
    def get_all_possible_names(self, type):
        """
        Return all possible names for type
        'goal'  - all unique object names
        'block' - all block names
        'agent' - all agent names
        """
        if type == "goal":
            return self.get_selected_goal_classes()
        else:
            return self.items[type].keys()

    def shuffle_classes(self, type):
        K = self.items[type].keys()
        V = self.items[type].values()
        random.shuffle(V)
        self.items[type].update(dict(zip(K, V)))

    def select_goal_classes(self):
        """
        Sample a number of classes (class_per_session) for interaction within a session
        """
        if self.class_per_session > 1:
            self.sel_classes = random.sample(self.items["goal"].keys(), self.class_per_session)
        else:
            self.sel_classes = self.items["goal"].keys()

    def get_selected_goal_classes(self):
        """
        Get the selected classes for a session
        """
        if not self.sel_classes:
            self.select_goal_classes()
        return self.sel_classes

    """
    def within_session_reinstantiation(self, e_list, step_list):
        # re-instantiate within the same session
        # re-load from map config with the same set of sampled classes
        assert len(e_list) == len(step_list)
        for e, l in zip(e_list, step_list):
            self.delete_entity(e)
            self.set_property(e, property_value_dict={"loc" : tsum(e.loc, l)})

    """
