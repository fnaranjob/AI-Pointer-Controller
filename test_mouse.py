import pyautogui

screenWidth, screenHeight = pyautogui.size()
pyautogui.moveTo(100,100,0.25)
pyautogui.moveTo(screenWidth-100,100,0.25)
pyautogui.moveTo(screenWidth-100,screenHeight-100,0.25)
pyautogui.moveTo(100,screenHeight-100,0.25)
pyautogui.moveTo(100,100,0.25)
print(screenHeight)
print(screenWidth)