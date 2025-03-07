// Copyright (C) 2019-present eyeo GmbH
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

"use strict";

// const getById = document.getElementById;
function getById(id)
{
  return document.getElementById(id);
}

function removeAllChildren(element)
{
  while (element.firstChild)
  {
    element.removeChild(element.firstChild);
  }
}

function makeElement(tag, params)
{
  const element = document.createElement(tag);
  if (params)
  {
    for (let [key, value] of Object.entries(params))
    {
      element[key] = value;
    }
  }
  return element;
}

function showMessage(type, message, hideDelay)
{
  const messageElement = makeElement("div", {
    className: type,
    textContent: message,
    onclick()
    {
      this.remove();
    }
  });
  getById("messages").append(messageElement);
  if (hideDelay)
  {
    setTimeout(() => messageElement.remove(), hideDelay * 1000);
  }
}

// Base class for ScreenshotsMode and DetectionsMode.
class Mode
{
  constructor(images)
  {
    this.images = images;
    this.imageSizes = [];
    this.imageSizeNo = 1;
    this.pageSize = 50;
    this.pageNo = 1;
  }

  get imageSize()
  {
    return this.imageSizes[this.imageSizeNo];
  }

  get maxPageNo()
  {
    return Math.floor(this.visibleImages.length / this.pageSize - 0.0001) + 1;
  }

  // Subclasses should override.
  filterImages()
  {
    return this.images;
  }

  get visibleImages()
  {
    // Cache filtered images for performance.
    const filters = this.showAll + "_" + this.showFP + "_" + this.showFN;
    if (this._lastFilters != filters)
    {
      this._visibleImages = this.filterImages();
      this._lastTypes = this.showAll + filters;
    }

    return this._visibleImages;
  }

  get currentPageImages()
  {
    const start = (this.pageNo - 1) * this.pageSize;
    const end = this.pageNo * this.pageSize;
    return this.visibleImages.slice(start, end);
  }

  magnify(image)
  {
    let magnified = getById("magnified");
    removeAllChildren(magnified);
    magnified.style = "display:block";
    magnified.onclick = ev => this.unmagnify();
    magnified.focus();
    magnified.appendChild(makeElement("img", {src: image.name}));
    return magnified;
  }

  unmagnify()
  {
    let magnified = getById("magnified");
    removeAllChildren(magnified);
    magnified.style = "display:none";
  }

  makeImg(image)
  {
    return makeElement("img", {
      src: image.name,
      style: "max-height: " + this.imageSize + "px;",
      onclick: ev => this.magnify(image)
    });
  }

  draw()
  {
    let images = this.currentPageImages.map(image => this.makeImg(image));
    const imagesElement = getById("images");
    removeAllChildren(imagesElement);
    images.map(img => imagesElement.appendChild(img));

    getById("page-no").innerHTML = this.pageNo;
    getById("prev").disabled = this.pageNo <= 1;
    getById("next").disabled = this.pageNo >= this.maxPageNo;

    if (this.maxPageNo > 1)
    {
      let pageSlider = getById("page-slider");
      pageSlider.title = "Page: " + this.pageNo;
      pageSlider.max = this.maxPageNo;
      pageSlider.value = this.pageNo;
      getById("pager").removeAttribute("style"); // To unhide.
    }
    else
    {
      getById("pager").style = "display:none";
    }

    let sizeSlider = getById("size-slider");
    sizeSlider.title = "Size: " + this.imageSize;
    sizeSlider.max = this.imageSizes.length - 1;
    sizeSlider.value = this.imageSizeNo;

    const modeButtons = document.getElementById("mode-buttons").children;
    for (let i = 0; i < modeButtons.length; i++)
    {
      const button = modeButtons[i];
      button.setAttribute("class",
                          button.id == "mode-" + this.name ? "" : "off");
    }
  }

  prevPage()
  {
    this.pageNo = Math.max(1, this.pageNo - 1);
    this.draw();
  }

  nextPage()
  {
    this.pageNo = Math.min(this.maxPageNo, this.pageNo + 1);
    this.draw();
  }

  pageChanged()
  {
    this.pageNo = getById("page-slider").value;
    this.draw();
  }

  sizeChanged()
  {
    this.imageSizeNo = getById("size-slider").value;
    this.draw();
  }

  typesChanged()
  {
    this.showAll = getById("all-cb").checked;
    this.showFP = getById("fp-cb").checked;
    this.showFN = getById("fn-cb").checked;
    this.pageNo = 1; // Reset page number so we don't end up on an empty one.
    this.draw();
  }

  activate()
  {
    this.typesChanged();
    this.pageNo = 1;
    this.imageSizeNo = 1;
    this.draw();
  }
}

class ScreenshotsMode extends Mode
{
  constructor(images)
  {
    super(images);
    this.imageSizes = [100, 200, 300, 400, 600, 900, 1200];
    this.pageSize = 50;
    this.name = "screenshots";
  }

  filterImages()
  {
    return this.images.filter(img =>
    {
      if (this.showAll)
        return true;
      if (this.showFP && img.fp > 0)
        return true;
      if (this.showFN && img.fn > 0)
        return true;
    });
  }

  makeImg(image)
  {
    const img = super.makeImg(image);
    img.setAttribute("class", "screenshot");
    return img;
  }
}

class DetectionsMode extends Mode
{
  constructor(images)
  {
    let all = [];
    images.map(image =>
    {
      for (let [type, boxes] of [["td", image.detections.true],
                                 ["fd", image.detections.false],
                                 ["dt", image.ground_truth.detected],
                                 ["mt", image.ground_truth.missed]])
      {
        boxes.map(
          box => all.push({name: box.file, type, origin: image, box: box.box})
        );
      }
    });
    super(all);
    this.imageSizes = [50, 100, 200, 400, 600, 900];
    this.pageSize = 250;
    this.name = "detections";
  }

  setSimilars(similars)
  {
    this.images.map(image => image.similars = similars[image.name]);
  }

  filterImages()
  {
    return this.images.filter(img =>
    {
      if (this.showAll)
        return true;
      if (this.showFP && img.type == "fd")
        return true;
      if (this.showFN && img.type == "mt")
        return true;
    });
  }

  makeImg(image)
  {
    const img = super.makeImg(image);
    img.setAttribute("class", image.type);
    return img;
  }

  magnify(image)
  {
    let magnified = super.magnify(image);
    if (image.box[2] - image.box[0] > image.box[3] - image.box[1])
    {
      // Horizontal image, would be better in two rows.
      magnified.appendChild(makeElement("br"));
    }
    magnified.appendChild(makeElement("img", {src: image.origin.name}));
    if (image.similars)
    {
      magnified.appendChild(makeElement("br"));
      magnified.appendChild(makeElement("h2", {
        textContent: "Similar fragments"
      }));
      image.similars.map(sim =>
      {
        magnified.appendChild(makeElement("img", {src: sim.file}));
      });
    }
  }
}

function loadData(path)
{
  let req = new XMLHttpRequest();
  return new Promise((resolve, reject) =>
  {
    req.onreadystatechange = e =>
    {
      if (req.readyState == XMLHttpRequest.DONE)
      {
        let status = req.status;
        if (status == 200)
        {
          if (req.responseType != "json")
            resolve(JSON.parse(req.response));
          else
            resolve(req.response);
        }
        else
          reject({status, statusText: req.statusText});
      }
    };
    req.open("GET", path);
    req.send();
  });
}

let modes = {};
let currentMode;

function switchMode(mode)
{
  if (currentMode)
    currentMode.unmagnify();
  currentMode = modes[mode];
  if (currentMode)
    currentMode.activate();
}

async function init()
{
  const dataPath = "data.json";
  const nnPath = "nn.json";

  try
  {
    const data = await loadData(dataPath);
    modes = {
      screenshots: new ScreenshotsMode(data),
      detections: new DetectionsMode(data)
    };
    switchMode("screenshots");
  }
  catch (e)
  {
    showMessage("error", "Can't load " + dataPath + ": " + e.statusText);
  }

  if (modes.detections)
  {
    try
    {
      const nns = await loadData(nnPath);
      modes.detections.setSimilars(nns);
    }
    catch (e)
    {
      if (e.statusText)
      {
        showMessage("warning", "Can't load " + nnPath + ": " + e.statusText +
                    ". Will proceed without similarity data.", 10);
      }
      else
      {
        showMessage("warning", "Similarity data from " + nnPath +
                    " has wrong format: " + e, 10);
      }
    }
  }
}
