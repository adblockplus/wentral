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

let pageSize = 50;
let imageSizes = [100, 200, 300, 400, 600, 900, 1200];
let imageSizeNo = 1;
let pageNo = 1;
let visibleImages = imageData;

function getMaxPage()
{
  return Math.floor(visibleImages.length / pageSize - 0.0001) + 1;
}

function getById(id)
{
  return document.getElementById(id);
}

function magnify(img)
{
  let src = img.src;
  let magnified = getById("magnified");

  magnified.style = "display:block";
  magnified.innerHTML = "<img src=\"" + src + "\" onclick=\"unmagnify()\"/>";
}

function unmagnify()
{
  getById("magnified").style = "display:none";
}

function makeImgTag(filename)
{
  let size = imageSizes[imageSizeNo];
  let style = "max-width: " + size + "px; max-height: " + size + "px;";
  return "<img style=\"" + style + "\" src=\"" + filename +
         "\" onclick=\"magnify(this)\" />";
}

function redraw()
{
  let pageStart = (pageNo - 1) * pageSize;
  let pageEnd = Math.min(pageStart + pageSize, visibleImages.length);
  let imgTags = [];

  for (let i = pageStart; i < pageEnd; i++)
  {
    imgTags.push(makeImgTag(visibleImages[i].name));
  }
  getById("images").innerHTML = imgTags.join("\n");

  getById("page-no").innerHTML = pageNo;
  getById("prev").disabled = pageNo <= 1;
  getById("next").disabled = pageEnd >= visibleImages.length;

  let pageSlider = getById("page-slider");
  pageSlider.title = "Page: " + pageNo;
  pageSlider.max = getMaxPage();
  pageSlider.value = pageNo;

  let sizeSlider = getById("size-slider");
  sizeSlider.title = "Size: " + imageSizes[imageSizeNo];
  sizeSlider.max = imageSizes.length - 1;
  sizeSlider.value = imageSizeNo;
}

function prevPage()
{
  pageNo = Math.max(1, pageNo - 1);
  redraw();
}

function nextPage()
{
  pageNo = Math.min(getMaxPage(), pageNo + 1);
  redraw();
}

function pageChanged()
{
  pageNo = getById("page-slider").value;
  redraw();
}

function sizeChanged()
{
  imageSizeNo = getById("size-slider").value;
  redraw();
}

function typesChanged()
{
  let showAll = getById("all-cb").checked;
  let showFP = getById("fp-cb").checked;
  let showFN = getById("fn-cb").checked;

  visibleImages = imageData.filter(img =>
  {
    if (showAll) return true;
    if (showFP && img.fp > 0) return true;
    if (showFN && img.fn > 0) return true;
  });

  pageNo = 1;
  redraw();
}

function init()
{
  typesChanged();
}
