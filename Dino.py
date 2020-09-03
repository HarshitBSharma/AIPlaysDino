from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64


game_url = "chrome://dino"
chrome_browser_path = ".//Driver/chromedriver.exe"
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

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
    def __init__(self):
        self.game = Game()
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
        

getbase64script = "canvasRunner = document.getElementById('runner-canvas');\
                    return canvasRunner.toDataURL().substring(22)"

def screenshot(driver):
    """image_b64 = driver.execute_script(getbase64script)
    screen = np.array(cv2.imread(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image"""
    filename = './/Screenshots/Screenshot.png'
    driver.save_screenshot(filename)
    image = cv2.imread(filename)
    image = process_img(image)
    return image

def process_img(image):
    #image = image[175:800, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image>255] = 255
    image = cv2.resize(84, 84)
    image = np.reshape(image, (84, 84, 1))
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


"""dino = Dinosaur()
dino.jump()
time.sleep(2)
screen = screenshot(dino.game.driver)
print(f"Image dimensions: {screen.shape}")
cv2.imshow('hah', screen)
cv2.waitKey(0)
dino.game.close_all()
#game_env = GameEnvironment(dino, game)"""

