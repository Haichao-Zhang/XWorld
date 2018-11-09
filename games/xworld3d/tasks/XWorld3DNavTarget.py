import random
from xworld3d_task import XWorld3DTask
from maze2d import print_env

"""
This file implements an xworld teaching task.
The task class contains several stage functions.

Each stage function must return three outputs:

next_stage  - the name of the next stage function
reward      - the reward of the current stage
sentence    - the sentence generated by the current stage

Finally, the function get_stage_names() return all the stage functions the
user wants to register.

This task asks the agent to go to a certain object.

Example:
Please go to the apple.
"""

class XWorld3DNavTarget(XWorld3DTask):
    def __init__(self, env):
        super(XWorld3DNavTarget, self).__init__(env)

    def idle(self):
        goals = self._get_goals()
        agent, _, _ = self._get_agent()
        targets = [g for g in goals if self._reachable(agent.loc, g.loc)]

        assert targets, "map too crowded?"
        sel_goal = random.choice(targets)

        ## set all goals that have the same name
        targets = [g for g in goals if g.name == sel_goal.name]

        self._record_target(targets);
        self._bind("S -> start")
        self._bind("G -> '" + sel_goal.name + "'")
        self.sentence = self._generate()
        return ["navigation_reward", 0.0, self.sentence]

    def navigation_reward(self):
        reward, time_out = self._time_reward()
        if not time_out:
            agent, _, _ = self._get_agent()
            objects_reach_test = [g.id for g in self._get_goals() \
                                  if self._reach_object(agent.loc, agent.yaw, g)]
            if [t for t in self.target if t.id in objects_reach_test]:
                reward = self._successful_goal(reward)
            elif objects_reach_test:
                reward = self._failed_goal(reward)
        return ["navigation_reward", reward, self.sentence]

    def get_stage_names(self):
        """
        return all the stage names; does not have to be in order
        """
        return ["idle", "navigation_reward"]

    def _define_grammar(self):
        all_goal_names = self._get_all_goal_names_as_rhs()
        grammar_str = """
        S --> start | timeup | correct | wrong
        start -> I0 | I1 | I2 | I3 | I4 | I5 | I6
        correct -> 'Well' 'done' '!'
        wrong -> 'Wrong' '!'
        timeup -> 'Time' 'up' '.'
        I0 -> G
        I1 -> A G 'please' '.'
        I2 -> 'Please' A G '.'
        I3 -> A G '.'
        I4 -> G 'is' 'your' D '.'
        I5 -> G 'is' 'the' D '.'
        I6 -> Y A G '?'
        A -> 'go' 'to' | 'navigate' 'to' | 'reach' | 'move' 'to' | 'collect'
        Y -> 'Could' 'you' 'please' | 'Can' 'you' | 'Will' 'you'
        D -> 'destination' | 'target' | 'goal' | 'end'
        G --> %s
        """ % all_goal_names
        return grammar_str, "S"
