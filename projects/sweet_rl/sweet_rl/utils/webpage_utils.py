from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver import FirefoxOptions
import time
import os
import traceback

# # Your HTML code snippet
# html_snippet = "<html> \n <head> \n <title>NTFS Disk Issue with TimeMachine</title> \n </head> \n <body> \n <h1>My NTFS Disk Suddenly Became Read-Only with TimeMachine</h1> \n <p>Gain write access to NTFS disks on MacOS by installing third-party drivers like Tuxera NTFS.</p> \n <p>Causes: TimeMachine attempted to clean the disk resulting in a read-only state.</p> \n <p>Solution Steps: </p> \n <ol> \n <li>Install a third-party NTFS driver, such as Tuxera NTFS.</li> \n <li>Use the driver to gain write access to the disk.</li> \n </ol> \n </body> \n </html>"

# # Save the HTML snippet to a temporary file
# with open("temp_page.html", "w") as file:
#     file.write(html_snippet)
import re
def replace_urls(text):
    # Regular expression to find the URLs
    pattern = r"https://source\.unsplash\.com/random/(\d+)x(\d+)/\?[\w=]+"
    # Function to replace each match with the new URL format
    def replace_match(match):
        width, height = match.groups()
        return f"https://picsum.photos/id/48/{width}/{height}"
    
    # Use re.sub to replace all occurrences in the text
    new_text = re.sub(pattern, replace_match, text)
    
    # Make sure that the new text has id 48 for all images
    # Define the regex pattern to match the URLs
    pattern = r'https://picsum\.photos/(\d+)/(\d+)'
    
    # Define the replacement pattern
    replacement = r'https://picsum.photos/id/48/\1/\2'
    
    # Use re.sub to replace all matches in the paragraph
    new_text = re.sub(pattern, replacement, new_text)
    
    return new_text


def get_driver():
    # Set up Chrome options
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.binary_location = "/home/yifeizhou/opt/google/chrome/firefox/firefox"
    # options.add_argument('--log-level=3')
    # service = Service(log_path=os.devnull)  # Redirect logs to nowhere
    # Set up the Firefox driver
    driver = webdriver.Firefox( options=options)
    return driver

    # chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--disable-dev-shm-usage")

    # # Set up the Chrome driver
    # service = Service(chrome_driver_path)  # Update with your path to chromedriver
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    # return driver


def render_full_html(driver, html_snippet, temp_path, env_id=0):
    # asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    # # Apply the nest_asyncio patch
    # nest_asyncio.apply()
    current_time = time.time()
    # image_path = os.path.join(temp_path, f"{env_id}_{current_time}.png")
    # import IPython; IPython.embed()
    # HTML(string=html_snippet).write_pdf(image_path)
    # return image_path
    # return render_html_and_capture_screenshot(html_snippet, )
    # try:
    #     # Try to get the current event loop
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     # If there is no event loop, create a new one
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    # return loop.run_until_complete(
    # render_html_and_capture_screenshot(html_snippet, os.path.join(temp_path, f"{env_id}_{current_time}.png"))
    # )
    
    html_file_path = os.path.join(temp_path, f"{env_id}_{current_time}.html")
    image_path = os.path.join(temp_path, f"{env_id}_{current_time}.png")
    # Save the HTML snippet to a temporary file
    with open(os.path.join(temp_path, f"{env_id}_{current_time}.html"), "w") as file:
        file.write(html_snippet)
    # imgkit.from_file(html_file_path, image_path)
    
    try:
        # Open the local HTML file
        driver.get(f"file://{html_file_path}")
        driver.get_full_page_screenshot_as_file(image_path)

        # # Wait for the page to load completely
        # time.sleep(1)  # Adjust the sleep time as needed

        # total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        # total_width = driver.execute_script("return document.body.parentNode.scrollWidth")
        # # # Set the window size to the dimensions of the page
        # driver.set_window_size(total_width, total_height)
        
        # time.sleep(1)
        # # Take a screenshot
        # driver.save_screenshot(image_path)
        os.remove(html_file_path)
        return image_path
    except Exception as e:
        print(e)
        traceback.print_exc()
        if os.path.exists(html_file_path):
            os.remove(html_file_path)
        return None

import re
def extract_html_snippet(paragraph):
    # Regular expression pattern to match the entire HTML content
    paragraph = replace_urls(paragraph)
    html_pattern = r'<html.*?>.*?</html>'
    
    # Search for the HTML snippet in the paragraph
    match = re.search(html_pattern, paragraph, re.DOTALL)
    
    if match:
        return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
    else:
        html_pattern = r'<body.*?>.*?</body>'
        match = re.search(html_pattern, paragraph, re.DOTALL)
        if match:
            return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
        else:
            return paragraph, None