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
        self.sel_classes = [] # selected classes for a session
        self.dis_classes = [] # selected distractor classes for a session
        self.shuffle = False # shuffle classes
        self.num_distractor = 1 # number of distractors
        # (y, x, z)
        self.nav_loc_set = [(2, 0, 0), (2, 4, 0), (4, 2, 0)]
        self.dia_loc_set = [(0, 0, 0), (0, 4, 0)]
        self.dis_loc_set = [] # distraction location

        self.num_distractor = min(self.num_distractor, len(self.nav_loc_set) - len(self.dia_loc_set))

        """
        random.shuffle(self.nav_loc_set)
        for i in range(self.num_distractor):
            loc = self.nav_loc_set.pop()
            self.dis_loc_set += [loc]
        """

        self.teach_loc = (0, 2, 0)
        self.agent_yaw_set = [3.14] # [0, 3.14] # yaw set for agent

    def _configure(self, select_class=True):
        self.set_goal_subtrees([ "others", "furniture"])
        ## these are all the object classes
        self.set_dims(5, 5)

        if select_class:
            # re-select goal class for a new session
            # self.sel_classes = self.select_goal_classes(self.class_per_session)
            goal_class = self.get_selected_goal_classes(reselect=True)
            dis_class = self.get_distractor_classes(reselect=True)
            # randomly partation the location set
            loc_set = self.nav_loc_set + self.dis_loc_set
            random.shuffle(loc_set)
            self.nav_loc_set = loc_set[0:-self.num_distractor]
            self.dis_loc_set = loc_set[-self.num_distractor:]

        if self.shuffle:
            self.shuffle_classes("goal")

        self.set_entity(type="agent", loc=(2, 2, 0))
        # self.set_entity(type="goal", loc=self.nav_loc_set[0])
        # self.set_entity(type="goal", loc=self.nav_loc_set[1])
        self.set_entity(type="goal", loc=self.dia_loc_set[0])
        self.set_entity(type="goal", loc=self.dia_loc_set[1])
        for i in range(self.num_distractor):
            self.set_entity(type="goal", loc=self.dis_loc_set[i])

        sel_goals = self.get_selected_goal_classes()
        dis_goals = self.get_distractor_classes()

        # sel_goals.pop()
        # sel_goals += ["carpet"]

        """
        random.shuffle(sel_goals)
        for i, e in enumerate(self.get_nav_goals()):
            self.set_property(e, property_value_dict={"name" : sel_goals[i], \
                                                      "yaw" : 2})
        """
        random.shuffle(sel_goals)
        for i, e in enumerate(self.get_dia_goals()):
            self.set_property(e, property_value_dict={"name" : sel_goals[i], \
                                                      "yaw" : 0})
        random.shuffle(dis_goals)
        for i, e in enumerate(self.get_dis_goals()):
            self.set_property(e, property_value_dict={"name" : dis_goals[i], \
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

    def get_dis_goals(self):
        goals = self.get_goals()
        dis_goals = [g for g in goals if g.loc in self.dis_loc_set]
        return dis_goals

    @overrides(XWorld3DEnv)
    def get_all_possible_names(self, type):
        """
        Return all possible names for type
        'goal'  - all unique object names
        'block' - all block names
        'agent' - all agent names
        """
        if type == "goal":
            return self.get_selected_goal_classes() + self.get_distractor_classes()
        else:
            return self.items[type].keys()

    def shuffle_classes(self, type):
        K = self.items[type].keys()
        V = self.items[type].values()
        random.shuffle(V)
        self.items[type].update(dict(zip(K, V)))

    def select_goal_classes(self, goal_num=-1, goal_exclusive_set=[]):
        """
        Sample a number of classes (class_per_session) for interaction within a session
        goal_num: number of goals to be selected; get all goals if <=0
        goal_exclusive_set: a set of goals to be excluded from before goal selection
        """
        all_goals = self.items["goal"].keys()
        valid_goals = list(set(all_goals) - set(goal_exclusive_set))
        if goal_num >= 1:
            sel_classes = random.sample(valid_goals, goal_num)
        else:
            sel_classes = valid_goals
        return sel_classes

    def get_selected_goal_classes(self, reselect=False):
        """
        Get the selected classes for a session
        """
        if not self.sel_classes or reselect:
            self.sel_classes = self.select_goal_classes(self.class_per_session)
        return self.sel_classes

    def get_distractor_classes(self, reselect=False):
        """
        Get the selected classes for a session
        """
        # should select the goal class first
        if not self.sel_classes:
            self.sel_classes = self.select_goal_classes(self.class_per_session)
        if not self.dis_classes or reselect:
            self.dis_classes = self.select_goal_classes(self.num_distractor, self.sel_classes)
        return self.dis_classes

    """
    def within_session_reinstantiation(self, e_list, step_list):
        # re-instantiate within the same session
        # re-load from map config with the same set of sampled classes
        assert len(e_list) == len(step_list)
        for e, l in zip(e_list, step_list):
            self.delete_entity(e)
            self.set_property(e, property_value_dict={"loc" : tsum(e.loc, l)})

    """
