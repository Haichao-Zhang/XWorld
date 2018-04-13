import random
from xworld3d_task import XWorld3DTask
from py_util import overrides
from py_util import tsum
import numpy as np

class XWorld3DDiaNav(XWorld3DTask):
    def __init__(self, env):
        super(XWorld3DDiaNav, self).__init__(env)
        self.max_steps = 20 # maximum number of steps, should be related to number of sel classes
        self.speak_correct_reward = 1
        self.speak_incorrect_reward = -1
        self.question_ask_reward = 0.1
        self.nothing_said_reward = -1

        self.reset_dialog_setting()
        ## some config para
        self.stepwise_reward = True
        self.success_reward = 1
        self.failure_reward = -0.1
        self.step_penalty = -0.01

        ## for teaching moving objects
        self.active_loc = None
        self.teach_step_max = 4
        self.teach_step_cur = 0

    def reset_dialog_setting(self):
        self.question_ratio = 0.5 # the chance of asking a question or making a statement
        self.teacher_sent_prev_ = [] # stores teacher's sentences in a session in order
        self.behavior_flags = []
        self.teach_step_cur = 0

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
            self.active_loc = random.choice(self.env.dia_loc_set)
            active_goals = [g for g in self.get_dia_goals() if g.loc == self.active_loc]
            active_goal = active_goals[0]
            return active_goal

    def idle(self):
        """
        Start a task
        """
        # print("--------idle ")
        self.teach_step_cur = 0
        self.task_type = self.get_task_type()
        agent, _, _ = self._get_agent()

        return ["move_or_teach", 0.0, ""]

        """
        if self.task_type == "dia":
            sel_goal = random.choice(self.get_dia_goals())
            self.dia_goal = sel_goal
            ## first generate all candidate answers
            self._bind("S -> statement")
            self._set_production_rule("G -> " + " ".join(["'" + sel_goal.name + "'"]))
            self.answers = self._generate_all()

            ## then generate the question
            self._bind("S -> question")
            self.questions = self._generate_all()

            sent = self.sentence_selection_with_ratio()
            self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
            teacher_sent = self._generate_and_save([sent])
            q_from_teacher = (teacher_sent == "" or teacher_sent in self.questions)
            if q_from_teacher: # dialog interaction
                return ["reward", 0.0, teacher_sent]
            else:
                return ["command", 0.0, teacher_sent]
        else:
            sel_goal = random.choice(self.get_nav_goals())
            ## first generate all candidate answers
            self._bind("S -> command")
            self._set_production_rule("G -> " + " ".join(["'" + sel_goal.name + "'"]))
            self.commands = self._generate_all()
            sent = random.choice(self.commands)
            self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
            teacher_sent = self._generate_and_save([sent])
            return ["reward", 0.0, teacher_sent]
        """

    def move_or_teach(self):
        """
        move object or teach the object
        """
        # print("--------idle ")
        self.task_type = self.get_task_type()
        agent, _, _ = self._get_agent()
        # move always, teach dependes

        # self.env.within_session_reinstantiation([self.active_goal], [self.loc_step])
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

        print("%s %s %s" % (l1, l2, diff))
        theta = np.arccos(np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2)))

        teacher_sent = ""
        if self.teach_step_cur < self.teach_step_max:
            self.teach_step_cur += 1
            print("theta %s" %(theta))
            if np.abs(theta) <= 1: # move and teach
                ## first generate all candidate answers
                self._bind("S -> statement")
                self._set_production_rule("G -> " + " ".join(["'" + active_goal.name + "'"]))
                self.answers = self._generate_all()

                sent = self.sentence_selection_with_ratio()
                self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
                teacher_sent = self._generate_and_save([sent])
            return ["move_or_teach", 0.0, teacher_sent]
        else:
            sel_goal = random.choice(self.get_nav_goals())
            ## first generate all candidate answers
            self._bind("S -> command")
            self._set_production_rule("G -> " + " ".join(["'" + sel_goal.name + "'"]))
            self.commands = self._generate_all()
            sent = random.choice(self.commands)
            self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
            teacher_sent = self._generate_and_save([sent])
            return ["command", 0.0, teacher_sent]

    def command(self):
        """
        Issue a command
        """
        # print("--------command ")
        agent, _, _ = self._get_agent()
        goals = self._get_goals()
        assert len(goals) > 0, "there is no goal on the map!"

        sel_goal = self.dia_goal
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
        self._record_target(targets);

        # get agent's sentence (response to previous sentence from teacher)
        _, agent_sent, _ = self._get_agent()
        return ["command_and_reward", 0.0, teacher_sent]

    def command_and_reward(self):
        """
        Issue a command and give reward based on arrival
        """
        # print("--------command and reward ")
        agent, _, _ = self._get_agent()
        goals = self._get_goals()
        assert len(goals) > 0, "there is no goal on the map!"
        teacher_sent_prev = self._get_last_sent()

        reward, time_out = self._time_reward()
        if not time_out:
            agent, _, _ = self._get_agent()
            objects_reach_test = [g.id for g in self._get_goals() \
                                  if self._reach_object(agent.loc, agent.yaw, g)]
            if [t for t in self.target if t.id in objects_reach_test]:
                reward = self._successful_goal(reward)
                return ["conversation_wrapup", reward, self.sentence]
            elif objects_reach_test:
                reward = self._failed_goal(reward)
        return ["command_and_reward", reward, self.sentence]

    def reward(self):
        """
        Giving reward to the agent
        """
        # print("--------reward ")
        def get_reward(reward, success=None):
            """
            Internal function for compute reward based on the stepwise_reward flag.
            reward is the current step reward
            success: None: not an ending step, True: success, False: failure
            """
            if self.stepwise_reward:
                return reward
            elif success is None:
                # only step_penalty for intermediate steps in non-stepwise rewarding case
                return self.step_penalty
            elif success is True: #final stage
                return self.success_reward
            elif success is False:
                return self.failure_reward

        # get agent's sentence (response to previous sentence from teacher)
        _, agent_sent, _ = self._get_agent()
        # get teacher's sentence
        prev_sent = self._get_last_sent()
        # if the previous stage is a qa stage
        is_reply_correct = agent_sent in self.answers
        is_nothing_said = agent_sent == ""
        # extend_step is for provding answer by teacher

        sel_goal = self.dia_goal

        # update answers
        self._bind("S -> statement") # first bind S to statement
        #self._bind("G -> '%s'" % sel_goal.name)
        self._set_production_rule("G -> " + " ".join(["'" + sel_goal.name + "'"]))
        self.answers = self._generate_all()

        self.steps_in_cur_task += 1
        # decide reward and next stage
        if qa_stage_prev:
            if is_question_asked:
                # reward feedback
                if not is_nothing_said:
                    reward = self.question_ask_reward
                else:
                    reward = self.nothing_said_reward
                    self.behavior_flags += [False]
                # sentence feedback (answer/statement)
                self._bind("S -> statement")
                #self._bind("G -> '%s'" % sel_goal.name)
                self._set_production_rule("G -> " + " ".join(["'" + sel_goal.name + "'"]))
                teacher_sent = self._generate_and_save()
                return ["command", reward, teacher_sent]
            elif is_reply_correct:
                self.behavior_flags += [True]
                reward = self.speak_correct_reward
                reward = get_reward(reward, all(self.behavior_flags))
                teacher_sent = ""
                return ["conversation_wrapup", reward, teacher_sent]
            else:
                self.behavior_flags += [False]
                reward = self.speak_incorrect_reward
                sent = self.sentence_selection_with_ratio()
                self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
                teacher_sent = self._generate_and_save([sent])
        else:
            # reward feedback for different cases
            if is_reply_correct: # repeat statement
                reward = 0
            elif is_nothing_said:
                reward = self.nothing_said_reward
            elif is_question_asked:
                reward = self.speak_incorrect_reward
            else:
                self.behavior_flags += [False]
                reward = self.speak_incorrect_reward
            # sentence feedback
            sent = self.sentence_selection_with_ratio()
            self._set_production_rule("R -> " + " ".join(["'" + sent + "'"]))
            teacher_sent = self._generate_and_save([sent])
        reward = get_reward(reward)
        return ["command", reward, teacher_sent]

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
                "command", "command_and_reward",
                "reward", "conversation_wrapup"]

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
