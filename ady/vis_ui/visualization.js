// This file is part of Ad Detect YOLO <https://adblockplus.org/>,
// Copyright (C) 2019-present eyeo GmbH
//
// Ad Detect YOLO is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as
// published by the Free Software Foundation.
//
// Ad Detect YOLO is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Ad Detect YOLO. If not, see <http://www.gnu.org/licenses/>.

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
      if (this.showAll) return true;
      if (this.showFP && img.fp > 0) return true;
      if (this.showFN && img.fn > 0) return true;
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
    imageData.map(image =>
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

  filterImages()
  {
    return this.images.filter(img =>
    {
      if (this.showAll) return true;
      if (this.showFP && img.type == "fd") return true;
      if (this.showFN && img.type == "mt") return true;
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
  }
}

const modes = {
  screenshots: new ScreenshotsMode(imageData),
  detections: new DetectionsMode(imageData)
};

let currentMode;

function switchMode(mode)
{
  if (currentMode)
  {
    currentMode.unmagnify();
  }
  currentMode = modes[mode];
  currentMode.activate();
}

function init()
{
  switchMode("screenshots");
}
