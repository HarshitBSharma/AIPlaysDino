from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


game_url = "chrome://dino"
chrome_browser_path = ".//Driver/chromedriver.exe"
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
generation_score = []

class Game():
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("start-maximized")
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self.driver = webdriver.Chrome(executable_path=chrome_browser_path, chrome_options=chrome_options)
        try:
            self.driver.get(game_url)
        except:
            print('Exception has been handled')
        self.driver.execute_script("Runner.config.ACCELERATION=0")
        self.driver.execute_script(init_script)
        

    def start_playing(self):
        self.driver.execute_script('Runner.instance_.playing=true')

    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def restart(self):
        return self.driver.execute_script("return Runner.instance_.restart()")

    def pause(self):
        return self.driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self.driver.execute_script("return Runner.instance_.play()")

    # This function will be called when we want to jump
    def press_up(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)

    # This function will be called when we want to duck
    def press_down(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)

    # This function will be called when we don't want to jump or duck
    def press_right(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_RIGHT)

    # To get the current score of our game
    def get_score(self):
        score_array = self.driver.execute_script('return Runner.instance_.distanceMeter.digits')
        # Scores are stored in the form '0, 1, 4', for a score of 14.
        score = ''.join(score_array)
        return int(score)

    # To get the highscore of our game
    def get_highscore(self):
        highscore_array = self.driver.execute_script('return Runner.instance_.distanceMeter.highscore')
        for i in range(len(highscore_array)):
            if highscore_array[i] == 0:
                break
            highscore_array = highscore_array[i:]
            highscore = ''.join(highscore_array)
            return int(highscore)

    # Closing browser
    def close_all(self):
        self.driver.close()
        self.driver.quit()
        try:
            os.system('cmd /c taskkill /F /IM chromedriver.exe')
        except:
            print('No tasks found!')


# Our Agent who controls the T-Rex
class Dinosaur():
    def __init__(self, game):
        self.game = game
        # Jump function is called to start the game
        self.jump()

    def jump(self):
        self.game.press_up()

    def duck(self):
        self.game.press_down()

    def do_nothing(self):
        self.game.press_right()

    
class GameEnvironment():
    def __init__(self, agent, game):
        self.agent = agent
        self.game = game
        self.display = show_img()
        self.display.__next__()

    def get_next_state(self, actions):
        score = self.dino.get_score()
        highscore = self.game.get_highscore()

        reward = 0.1
        is_over = False
        if actions[0] == 1:
            self.agent.jump()
        elif actions[1] == 1:
            self.agent.duck()
        elif actions[2] == 1:
            self.agent.do_nothing()

        image = screenshot(self.game.driver)
        self.display.send(image)

        if self.agent.is_crashed():
            generation_score.append(score)
            time.sleep(0.1)
            self.game.restart()
            reward = -1
            is_over = True

        image = image_to_tensor(image)

        return image, reward, is_over, score, highscore

        

def screenshot(driver):
    filename = './/Screenshots/Screenshot.png'
    driver.save_screenshot(filename)
    image = cv2.imread(filename)
    image = process_img(image)
    return image

def process_img(image):
    #image = image[175:800, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image>255] = 255
    image = cv2.resize(image, (84, 84))
    image = np.reshape(image, (1, 84, 84))
    return image

def show_img():
    while True:
        screen = (yield)
        window_title = 'Dino Agent'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

def image_to_tensor(image):
    print(f"Before Transpose: {image.shape}")
    # image = np.transpose(image, (2, 0, 1))
    #print(f"After Transpose: {image.shape}")
    image_tensor = image.astype(np.float32)
    image_tensor = torch.from_numpy(image)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor


dinosaur = Dinosaur()
dinosaur.jump()
screen = screenshot(dinosaur.game.driver)
screen = np.reshape(screen, (84, 84, 1))
print(f"Screen shape is {screen.shape}")
"""cv2.imshow("ha", screen)
cv2.waitKey(0)"""

# Starting to Build our DQN
class DinoNetwork(nn.Module):
    def __init__(self):
        super(DinoNetwork, self).__init__()
        self.number_of_actions = 3
        self.gamma = 0.99
        self.initial_epsilon = 0.1
        self.final_epsilon = 0.0001
        self.number_of_iterations = 10000
        self.replay_memory_size = 1000
        self.minibatch_size = 1 

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(256)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 3)
        )


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        print(f"After last conv layer, shape is {x.shape}")
        x = x.flatten(start_dim=1, end_dim=-1)
        print(f"After Flattening shape is {x.shape}")
        x = self.fc1(x)
        x = self.fc2(x) 
        return x


screen = image_to_tensor(screen)
screen = torch.reshape(screen, (1, 1, 84, 84))
screen = screen.type(torch.cuda.FloatTensor)
model = DinoNetwork()
model.cuda()
output = model.forward(screen)
print(output.shape)

def train(model, start):
    optimizer = optim.Adam(model.paramaters(), lr=1e-4)
    criterion = nn.MSELoss()

    game = Game()
    dino = Dinosaur(game)
    game_state = GameEnvironment(dino, game)

    replay_memory = []

    actions = torch.zeros([model.number_of_actions], dtype=torch.cuda.float32)
    action[0] = 1

    image_data, reward, terminal, score, high_score = game_state.get_next_state(action)
    
    epsilon = model.initial_epsilon
    iteration = 0    

    epsilon decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations) 
    while iteration < model.number_of_iterations:
        output = model(image_data)
        
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        action = transfer_to_cuda(action)
        
        random_action = random.random() <= epsilon
        action_index = [torch.randint(model.number_of_actions, torch.size([]), dtype = torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        action_index = transfer_to_cuda(action_index)

        # Setting the action to 1 because you gotta jump first to start the game
        action[action_index] = 1

        image_data_1, reward, terminal, score, high_score = game_state.get_next_state(action)
        

        











def transfer_to_cuda(dummy_tensor):
    if torch.cuda.is_available():
        dummy_tensor.cuda()
    return dummy_tensor

    