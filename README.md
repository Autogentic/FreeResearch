## Use case

This project aims to bring a version of OpenAI Deep Research, Google Deep Research, Perplexity Deep Research, and Grok Deep Search to the public. entirely free and with unlimited use.

All you need is a free Google AI Studio API, which you can easily acquire.

It leverages an agentic framework with multiple LLMs to analyze and search the web for the information you seek. It conducts in-depth searches, exploring multiple directions based on the initial topic to maximize the value of the results.

Currently in Alpha, we welcome testers and contributors to help improve the project.

If you have access to paid deep research tools, a comparative analysis and benchmarking against them would be highly valuable.

## How to install

What you need:
Python 3.11 or higher - (https://www.python.org/downloads/ or https://anaconda.com/download/success)
Node.js - (https://nodejs.org/en)

1. Open a terminal(command prompt)
2. Choose a path on your computer where you want to download the program and run this:
```sh
git clone https://github.com/Autogentic/FreeResearch.git
```
3. Create an environment(I perfer anaconda): 
  - If you use anaconda:
```sh
conda create --name FreeResearch python=3.11.11
conda activate FreeResearch
```
  - If you use python venv:
      - ```sh
        python3.11 -m venv FreeResearch
        ```
      - Mac/Linux:
        ```sh
        source FreeResearch/bin/activate
        ```
      - Windows:
        ```sh
        FreeResearch\Scripts\activate.bat
        ```
4. Install python packages:
    ```sh
    pip install -r requirements.txt
    ```

5. Remove ".placeholder" from the following file: .env.placeholder (should be .env when finished)

6. Get you API key from https://aistudio.google.com/
    - Log in
    - Create API key (should be top left)
    - Place the generated API key inside the .env after the = sign. 
    - Should look something like: GEMINI_API_KEY=<ACTUAL API KEY>

Now you can run start using it from the CLI by just running: python freeresearch_core.py

7. If you want to use the frontend. Please continue with the following steps
   ```sh
    python backend_server.py
   ```
8. Open another terminal and install npm dependencies:
   cd Path/To/Project/FreeResearch
   ```sh
    npm install
    ```
9. start front end:
    ```sh
    npm start
    ```
10. Open the url on a browser:
  - 0.0.0.0:300

## Logical Flow
![pako_eNqtWOtu2zYUfhVCQxAMk1v5GtsICuTSFNncy-ys2Fb3By0d20Ik0SWpJG7TX_u_PcJ-7AH2d8-zF9geYYciKUqOnTTAFCA2z-U7F51zSPqTF7IIvKG3t_dpmhESZ7EckuIrIfvzhF2HS8rlfklD6lKmyYjOIBFIljwHv2SFOb8CpO6nLGOSZfDjvmNGMKd5IseQRcCBK7GILjhURDjNLicrGsbZAtnNIHCsDP2ssTT](https://github.com/user-attachments/assets/4cbc3629-33c7-4645-a35d-d15ef326287a)

## Disclaimer

This project is intended for educational purposes only. Users are responsible for ensuring compliance with any target website's terms of service, applicable laws, and ethical considerations. For added privacy, consider using a VPN. The author disclaims all liability for any misuse, damages, or consequences arising from the use of this code. Use at your own risk.


