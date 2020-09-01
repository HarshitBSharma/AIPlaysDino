from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import os


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

    def press_up(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)

    def press_down(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score = self.driver.execute_script('return Runner.instance_.distanceMeter.digits')
        score = ''.join(score)
        print(score)

    def close_all(self):
        self.driver.close()
        self.driver.quit()
        """"try:
            os.system('cmd /c taskkill /F /IM chromedriver.exe')
        except:
            print('No tasks found!')"""

g = Game()
g.start_playing()
g.press_up()
g.get_score()
time.sleep(2)
g.press_down()
g.get_score()
time.sleep(2)
g.get_score()
g.close_all()
