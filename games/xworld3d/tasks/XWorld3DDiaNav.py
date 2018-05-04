import random
from xworld3d_task import XWorld3DTask
from py_util import overrides
from py_util import tsum
import numpy as np

class XWorld3DDiaNav(XWorld3DTask):
    def __init__(self, env):
        super(XWorld3DDiaNav, self).__init__(env)
        self.max_steps = 50 # maximum number of steps, should be related to number of sel classes
        self.speak_correct_reward = 1
        self.speak_incorrect_reward = -1
        self.question_ask_reward = 0.1
        self.nothing_said_reward = -1

        self.reset_dialog_setting()
        ## some config para
        self.stepwise_reward = True
        self.success_reward = 1
        self.failure_reward = -1
        self.step_penalty = -0.05

        ## for teaching moving objects
        self.active_loc = None
        self.teach_step_max = 4
        self.teach_step_cur = 0
        self.taught_goal_loc = [] # loc for already taught goals

    def reset_dialog_setting(self):
        self.teacher_sent_prev_ = [] # stores teacher's sentences in a session in order
        self.behavior_flags = []
        self.teach_step_cur = 0
        self.taught_goal_loc = [] # loc for already taught goals

    def get_nav_goals(self):
        goals = self._get_goals()
        nav_goals = [g for g in goals if g.loc in self.env.nav_loc_set]
        return nav_goals

    def get_dia_goals(self):
        goals = self._get_goals()
        dia_goals = [g for g in goals if g.loc in self.env.dia_loc_set]
        return dia_goals

    def get_active_goal(self):
        # if the goal is on teach loc, then it is active
        # otherwise randomly select one object to teach
        ag = [g for g in self._get_goals() if g.loc == self.env.teach_loc]
        if len(ag) > 0:
            return ag[0]
        else:
            self.active_loc = random.choice(list(set(self.env.dia_loc_set) - set(self.taught_goal_loc)))
            self.taught_goal_loc.append(self.active_loc)

            active_goals = [g for g in self.get_dia_goals() if g.loc == self.active_loc]
            active_goal = active_goals[0]
            return active_goal

    def idle(self):
        """
        Start a task
        """
        # print("--------idle ")
        self.reset_dialog_setting()
        self.task_type = self.get_task_type()
        agent, _, _ = self._get_agent()

        return ["move_or_teach", 0.0, ""]

    def move_or_teach(self):
        """
        move object or teach the object
        """
        # print("--------idle ")
        self.task_type = self.get_task_type()
        agent, _, _ = self._get_agent()
        # move always, teach dependes
        teacher_sent = ""
        if self.teach_step_cur < self.teach_step_max:
            self.teach_step_cur += 1
            active_goal = self.get_active_goal()
            self.dia_goal = active_goal
            if self.active_loc != self.env.teach_loc:
                self.org_loc = self.active_loc
                self.active_loc = self.env.teach_loc
                self._move_entity(active_goal, self.env.teach_loc)
            else:
                self._move_entity(active_goal, self.org_loc)

            # check if teaching condition is satisfied
            l1 = np.array(active_goal.loc)
            # value of l1 is based on whether the env update happens next step or not
            l2 = np.array(agent.loc)
            diff = l1 - l2
            north_dir = np.array(tsum(*self.env.dia_loc_set)) / 2

            theta = np.arccos(np.dot(diff, north_dir) / (np.linalg.norm(diff) * np.linalg.norm(north_dir)))
            if np.abs(theta - 1.57) <= 0.5: # move and teach when obj in view
                ## first generate all candidate answers
                self._bind("S -> statement")
                self._set_production_rule("G -> " + " ".join(["'" + active_goal.name + "'"]))
                self.answers = self._generate_all()

                sent = self.sentence_selection_with_ratio()
                self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
                teacher_sent = self._generate_and_save([sent])
            return ["move_or_teach", 0.0, teacher_sent]
        else: # issue an navigation command in the end
            sel_goal = random.choice(self.get_nav_goals())
            ## first generate all candidate answers
            self._bind("S -> command")
            self._set_production_rule("G -> " + " ".join(["'" + sel_goal.name + "'"]))
            self.commands = self._generate_all()
            sent = random.choice(self.commands)
            self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
            teacher_sent = self._generate_and_save([sent])
            self.sentence = teacher_sent
            ## find all goals that have the same name as the just taught one but at a different location
            targets = [g for g in self.get_nav_goals() if g.name == sel_goal.name]
            self._record_target(targets)
            return ["command_and_reward", 0.0, teacher_sent]

    def command_and_reward(self):
        """
        Issue a command and give reward based on arrival
        All rewards are within [-1, 1]
        """
        agent, _, _ = self._get_agent()
        goals = self._get_goals()
        assert len(goals) > 0, "there is no goal on the map!"
        teacher_sent_prev = self._get_last_sent()

        reward, time_out = self._time_reward()
        reward = self.step_penalty # over-write the step penalty
        if not time_out:
            agent, _, _ = self._get_agent()
            objects_reach_test = [g.id for g in self._get_goals() \
                                  if self._reach_object(agent.loc, agent.yaw, g)]
            if [t for t in self.target if t.id in objects_reach_test]:
                # add with time penalty within _successful_goal function
                reward = self._successful_goal(reward)
                reward = np.clip(reward, self.failure_reward, self.success_reward)
                return ["conversation_wrapup", reward, self.sentence]
            elif objects_reach_test:
                reward = self._failed_goal(reward)
                reward = np.clip(reward, self.failure_reward, self.success_reward)
            return ["command_and_reward", reward, self.sentence]
        else:
            return ["conversation_wrapup", reward , self.sentence]

    @overrides(XWorld3DTask)
    def _reach_object(self, agent, yaw, object):
        collisions = self._parse_collision_event(self.env.game_event)
        theta, _, _ = self._get_direction_and_distance(agent, object.loc, yaw)
        return abs(theta) < self.orientation_threshold and object.id in collisions

    @overrides(XWorld3DTask)
    def _time_reward(self):
        # reward = XWorld3DTask.time_penalty
        reward = self.step_penalty
        self.steps_in_cur_task += 1
        h, w = self.env.get_dims()
        if self.steps_in_cur_task >= self.max_steps:
            self._record_failure()
            self._bind("S -> timeup")
            self.sentence = self._generate()
            self._record_event("time_up")
            reward += self.failure_reward
            return (reward, True)
        return (reward, False)

    @overrides(XWorld3DTask)
    def _successful_goal(self, reward):
        self._record_success()
        self._record_event("correct_goal")
        reward += self.success_reward
        self._bind("S -> correct")
        self.sentence = self._generate()
        return reward

    @overrides(XWorld3DTask)
    def _failed_goal(self, reward):
        self._record_failure()
        self._record_event("wrong_goal")
        reward += self.failure_reward
        self._bind("S -> wrong")
        self.sentence = self._generate()
        return reward

    @overrides(XWorld3DTask)
    def conversation_wrapup(self):
        """
        This dummpy stage simply adds an additional time step after the
        conversation is over, which enables the agent to learn language model
        from teacher's last sentence.
        """
        # print("--------wrapup ")
        if all(self.behavior_flags):
            self._record_success()
            self._record_event("correct_reply", next=True)
        else:
            self._record_failure()
            self._record_event("wrong_reply", next=True)
        self._record_event(self.prev_event)
        self.prev_event = None
        self.reset_dialog_setting()
        return ["idle", 0, ""]

    def get_stage_names(self):
        """
        return all the stage names; does not have to be in order
        """
        return ["idle", "move_or_teach",
                "command_and_reward",
                "conversation_wrapup"]

    def _define_grammar(self):
        if False:
            return self.get_sentence_level_grammar()
        else:
            return self.get_word_level_grammar()

    def get_sentence_level_grammar(self):
        grammar_str = """
        S --> question | statement
        question -> E | Q
        statement-> A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8
        E -> ''
        Q -> Q1 | Q2 | Q3
        Q1 -> 'what'
        Q2 -> 'what' M
        Q3 -> 'tell' 'what' N
        M ->  'is' 'it' | 'is' 'this' | 'is' 'there' | 'do' 'you' 'see' | 'can' 'you' 'see' | 'do' 'you' 'observe' | 'can' 'you' 'observe'
        N -> 'it' 'is' | 'this' 'is' | 'there' 'is' | 'you' 'see' | 'you' 'can' 'see' | 'you' 'observe' | 'you' 'can' 'observe'
        A1 -> G
        A2 -> 'it' 'is' G
        A3 -> 'this' 'is' G
        A4 -> 'there' 'is' G
        A5 -> 'i' 'see' G
        A6 -> 'i' 'observe' G
        A7 -> 'i' 'can' 'see' G
        A8 -> 'i' 'can' 'observe' G
        G  -> 'dummy'
        """
        return grammar_str, "S"

    def get_word_level_grammar(self):
        grammar_str = """
        S --> statement | command | correct | wrong | timeup
        statement-> G
        command -> C
        G -> 'dummy'
        C -> 'go' 'to' G
        correct -> 'correct'
        wrong -> 'wrong'
        timeup -> 'time' 'up'
        """
        return grammar_str, "S"

    def sentence_selection_with_ratio(self):
        return random.choice(self.answers)

    def _generate_and_save(self, teacher_sent = []):
        """
        generate (if teacher_sent is empty)  and save the teacher's sentence
        to teacher's previous sentence pool
        """
        if not teacher_sent:
            teacher_sent = [self._generate()]
        self.teacher_sent_prev_ = self.teacher_sent_prev_ + teacher_sent
        return teacher_sent[0]

    def _get_last_sent(self):
        """
        get the sentence from teacher in the last time step
        """
        assert self.teacher_sent_prev_, "make sure the previous sentence set is non-empty"
        sent = self.teacher_sent_prev_[-1]
        return sent

    def get_task_type(self):
        """
        get the type of the task according to the yaw of the agent
        this should only be called at he beginning of the game
        """
        a, _, _ = self._get_agent()
        if a.yaw == 0: # facing two objects initially
            return "nav" # task to be setup
        else:
            return "dia"
