import llm_prompt
import llm_utils
import ast

wrapper_template = """
class {}Wrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
    
    def reset(self, **kwargs):
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0

        num_item = info["inventory"]["{}"]
        if num_item > self.prev_count:
            reward += 1000
            # done = True
        self.prev_count = num_item

        return obs, reward, done, info
"""

navigation_wrapper_template = """
class {}NavigationWrapper(gym.Wrapper):

    def __init__(self, env, obj_index={}):
        super().__init__(env)
        self.target_obj = obj_index 

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        for i in range(left_index, right_index, 1):
            for j in range(up_index, down_index, 1):
                if (info['semantic'][i][j] == self.target_obj):
                    reward = 1000
                    done = True
                    return obs, reward, done, info
        
        return obs, reward, done, info
"""

wrapper_with_navigation_template = """
class {}WrapperWithNavigation(gym.Wrapper):

    def __init__(self, env, navigation_model, obj_index={}):
        super().__init__(env)
        self.model = navigation_model
        self.prev_count = 0
        self.obj_index = obj_index

    def reset(self, **kwargs):

        self.prev_count = 0
        
        obs = self.env.reset(**kwargs)

        valid = False

        for _ in range(100):

            if not valid:

                action, _  = self.model.predict(obs, deterministic=True)
                obs, _, _, info = self.env.step(action)

                player_pos = info['player_pos']

                left_index = max(0, player_pos[0] - 4)
                right_index = min(64, player_pos[0] + 4)
                up_index = max(0, player_pos[1] - 3)
                down_index = min(64, player_pos[1] + 3)
                
                for i in range(left_index, right_index, 1):
                    if not valid:
                        for j in range(up_index, down_index, 1):
                            if (info['semantic'][i][j] == self.obj_index):
                                valid = True
                                break
        return obs
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0
        num_item = info["inventory"][obj_name]

        if num_item > self.prev_count
            reward += 1000
            done = True
        self.prev_count = num_item

        return obs, reward, done, info
"""

def define_training_wrapper(obj_name):
    
    return wrapper_template.format(obj_name, obj_name)

def define_navigation_wrapper(obj_name, obj_index):

    return navigation_wrapper_template.format(obj_name, obj_index)

def define_training_wrapper_with_navigation(obj_name, obj_index):

    return wrapper_with_navigation_template.format(obj_name, obj_index)


def plan(tasks_list, num_step, rules):

    return plan_aux(tasks_list, [], 0, num_step, rules)


def plan_aux(tasks_list, command_list, current_step, num_step, rules):

    if current_step == num_step or len(tasks_list) == 0:
        return tasks_list, command_list

    current_tasks_list = tasks_list

    tasks_list = []
    command_list = []

    for subgoal in current_tasks_list:

        PLANNING_PROMPT = llm_prompt.compose_planning_prompt(rules)

        response = llm_utils.llm_chat(prompt="Current goal: " + subgoal, system_prompt=PLANNING_PROMPT, model="deepseek-chat")
        subgoals_list = []
        try: 
            llm_subgoals_list = ast.literal_eval(response)
            for new_subgoal in llm_subgoals_list:
                response = llm_utils.llm_chat(prompt=new_subgoal,system_prompt=llm_prompt.TRANS_PROMPT, model="deepseek-chat")
                if "None" not in response and response not in command_list:
                    tasks_list.append(new_subgoal)
                    command_list.append(response)

        except Exception as e:
            pass

    print(tasks_list)
    print(command_list)

    return plan_aux(tasks_list, command_list, current_step+1, num_step, rules)


if __name__ == "__main__":

    rules = open("rules.txt", 'r').read()
    tasks_list = ["place a furnace"]
    (tasks_list, object_list) = plan(tasks_list=tasks_list, num_step=3, rules=rules)

    for object in object_list:

        print(define_wrapper(object))
