# Photoshop-Shoreline-Toolkit-Manga-Typesetting
A Photoshop plugin that allows typesetters to use Gemini-2.5-flash-image and FLUX Kontext Dev (Together.ai) API.
Fits for team member management.

## Install

**Individual Translators:**
Run ```image_alignment_server.py``` first.
Then throw all *other files* to your Photoshop ```Plug-ins``` folder.
Activate it by clicking ```Plug-ins >>> Shoreline Toolkit >>> Shoreline Tools``` in Photoshop, then you'll see the panel.
Generate an API key for yourself by visiting ```127.0.0.1:8081```.
Type the endpoint ```http://127.0.0.1:8080``` and API key to your plugin settings pop-up.

**Team leader:**
Throw image_alignment_server.py on your Linux server and run it by:
```
python3 image_alignment_server.py
```
Visit ```[server address]:8081``` to generate API keys for your typesetter staff and set a expiration time, remember to change ```host, admin_host``` to ```0.0.0.0``` in ```config.json``` before first time launching the server.
Leader needs to pay for all members' usage since the server only uses the leader's balance account, but there is a usage calculation so you can share the bill.
Let your staff use the endpoint ```[server address]:8080``` in their plugin settings.

**Typesetter members:**
Throw all *other files* to your Photoshop ```Plug-ins``` folder.
Activate it by clicking ```Plug-ins >>> Shoreline Toolkit >>> Shoreline Tools``` in Photoshop, then you'll see the panel.
You can use your own Gemini or Together.ai APIs, but auto-alignment function is only usable when you are using the paired server endpoint above.

## Usage

### Image Output:

This plugin automatically detects whether the current document is close to "Black & White", then outputs a greyscale PNG, otherwise a colored JPEG.

### Image Editing:

The default prompt is: ```extend the content to the border, remove Japanest text.```

That is because the server adds a gray border to give some room for later aligning. But you can use a different prompt for more creative tasks (like guessing the missing content in a certain area?) in plugin settings.

The plugin prefers a RECTANGLE SELECTION.

After selecting, you can submit the image, and wait till it finishes or just do some other things.

Remember to apply the result by pressing the button.