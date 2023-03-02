# automate-lighthouse-reports

Uses Google's lighthouse (https://github.com/GoogleChrome/lighthouse) to build a set of reports from the URL list you pass into the configuration file.

It's a Node.JS script, so you need Node / NPM installed on your machine.

# Setup

After cloning the repo, run 

`npm install`

to install the dependencies.

In `config.json`, edit the following fields:

| Field | Example | Description |
|-------|---------|-------------|
| `url` | `["https://www.reddit.com/", "https://www.amazon.com/"]` | Array with list of fully formatted URLs to audit. |
| `lighthouseFlags` | `{"output": "csv", "disableDeviceEmulation": true}` | List of flags to pass to lighthouse. Full list available here: https://github.com/GoogleChrome/lighthouse/blob/8f500e00243e07ef0a80b39334bedcc8ddc8d3d0/typings/externs.d.ts#L52 |
| `chromeFlags` | `["--headless"]` | List of flags to pass to the Chrome launcher. Full list available here: https://peter.sh/experiments/chromium-command-line-switches/ |
| `writeTo` | `"/"` | The path where to write the reports - the tool will create the path if it doesn't exist. Remember the trailing slash in the end. |
| `sortByDate` | `true` | If `true`, stores the report in a folder structure of `writeTo`/year/month/url_1.csv, for example. If set to `false`, sorts by file, so `writeTo`/url_1/year/month/url_1.csv. |

# Run

Once you've set it up, you can run the audit tool with

`node script.js`

The process will be logged into the console. 

The reports will be written in the format you chose for the `output` key in the configuration, and they will be written in the folder structure you specified in the `writeTo` and `sortByDate` keys of the configuration file.
