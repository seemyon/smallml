const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const fs = require("fs");

const configJson = JSON.parse(fs.readFileSync("config.json"));

function launchChromeAndRunLighthouse(url, opts, config = null) {
  return chromeLauncher.launch({chromeFlags: opts.chromeFlags}).then(chrome => {
    console.log("Launching lighthouse for " + url);
    opts.lighthouseFlags.port = chrome.port;
    return lighthouse(url, opts.lighthouseFlags, config).then(res => {
      console.log("Parsing report for " + url);
      return chrome.kill().then(() => res.report);
    });
  });
}

configJson.url.forEach(address => {
  console.log("Starting analysis on " + address);
  launchChromeAndRunLighthouse(address, configJson).then(results => {
    const file = address.replace(/^https?:\/\//, "").replace(/[./]/g, "_") + "." + configJson.lighthouseFlags.output;
    const today = new Date();
    const folder = configJson.sortByDate ? configJson.writeTo + today.getFullYear() + "/" + (today.getMonth() + 1) + "/" : configJson.writeTo + file.replace("." + configJson.lighthouseFlags.output, "") + "/" + today.getFullYear() + "/" + (today.getMonth() + 1) + "/";
    const dest = folder + file;

    console.log("Writing analysis to " + dest);
    fs.mkdir(folder, { recursive: true }, (err) => {
      if (err) { console.log(err); }
      else {
        fs.writeFile(dest, results, (err) => {
          if (err) { console.log(err); }
          else { console.log("Analysis saved to " + dest) }
        });
      }
    });
  }).catch(err => console.log(err));
});