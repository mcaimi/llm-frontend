#!/usr/bin/env python

# load libraries
import sys
try:
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    import starlette.status as status
    from libs.ui import Ui, GRADIO_CUSTOM_PATH
except Exception as e:
    print(f"Caught exception: {e}")
    sys.exit(-1)

# start app
try:
    web_ui = Ui()
    web_ui.buildUi()
except Exception as e:
    print(f"Exception {e} caught: exiting")
    exit(1)

# build the application object
sd_app = FastAPI()

# add a root path
@sd_app.get("/")
async def get_root():
    # Redirect to the main Gradio App
    return RedirectResponse(url=GRADIO_CUSTOM_PATH, status_code=status.HTTP_302_FOUND)

# attach gradio app
web_ui.registerFastApiEndpoint(sd_app)

