let canvas = document.getElementById("drawing-board");
let ctx = canvas.getContext("2d");
let eraser = document.getElementById("eraser");
let brush = document.getElementById("brush");
let reSetCanvas = document.getElementById("clear");
let aColorBtn = document.getElementsByClassName("color-item");
let save = document.getElementById("save");
let undo = document.getElementById("undo");
let range = document.getElementById("range");
let scan = document.getElementById("scan");
let isRequesting = false

let clear = false;
let activeColor = 'black';
let lWidth = 4;

let historyDeta = [];


autoSetSize(canvas);

setCanvasBg('white');

listenToUser(canvas);

getColor();

//window.onbeforeunload = function(){
//    return "Reload site?";
//};

function autoSetSize(canvas) {
    canvasSetSize();

    function canvasSetSize() {
        let pageWidth = document.documentElement.clientWidth;
        let pageHeight = document.documentElement.clientHeight;

        canvas.width = pageWidth;
        canvas.height = pageHeight;
    }

    window.onresize = function () {
        canvasSetSize();
    }
}

function setCanvasBg(color) {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
}

function listenToUser(canvas) {
    console.log("listenToUser")
    let painting = false;
    let lastPoint = {x: undefined, y: undefined};

    if (document.body.ontouchstart !== undefined) {
        canvas.ontouchstart = function (e) {
            this.firstDot = ctx.getImageData(0, 0, canvas.width, canvas.height);//在这里储存绘图表面
            saveData(this.firstDot);
            painting = true;
            let x = e.touches[0].clientX;
            let y = e.touches[0].clientY;
            lastPoint = {"x": x, "y": y};
            ctx.save();
            drawCircle(x, y, 0);
        };
        canvas.ontouchmove = function (e) {
            if (painting) {
                let x = e.touches[0].clientX;
                let y = e.touches[0].clientY;
                let newPoint = {"x": x, "y": y};
                drawLine(lastPoint.x, lastPoint.y, newPoint.x, newPoint.y);
                lastPoint = newPoint;
            }
        };

        canvas.ontouchend = function () {
            painting = false;
        }
    } else {
        canvas.onmousedown = function (e) {
            this.firstDot = ctx.getImageData(0, 0, canvas.width, canvas.height);//在这里储存绘图表面
            saveData(this.firstDot);
            painting = true;
            let x = e.clientX;
            let y = e.clientY;
            lastPoint = {"x": x, "y": y};
            ctx.save();
            drawCircle(x, y, 0);
        };
        canvas.onmousemove = function (e) {
            if (painting) {
                let x = e.clientX;
                let y = e.clientY;
                let newPoint = {"x": x, "y": y};
                drawLine(lastPoint.x, lastPoint.y, newPoint.x, newPoint.y, clear);
                lastPoint = newPoint;
            }
        };

        canvas.onmouseup = function () {
            painting = false;
        };

        canvas.mouseleave = function () {
            painting = false;
        }
    }
}

function drawCircle(x, y, radius) {
    console.log("drawCircle")
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (clear) {
        ctx.clip();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    }
}

function drawLine(x1, y1, x2, y2) {
    console.log("drawLine")
    ctx.lineWidth = lWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (clear) {
        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.closePath();
        ctx.clip();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    } else {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.closePath();
    }
}

range.onchange = function () {
    lWidth = this.value;
};

eraser.onclick = function () {
    clear = true;
    this.classList.add("active");
    brush.classList.remove("active");
};

brush.onclick = function () {
    clear = false;
    this.classList.add("active");
    eraser.classList.remove("active");
};

reSetCanvas.onclick = function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setCanvasBg('white');
    var result_text = document.getElementById("info");
    result_text.innerHTML = "Result"
};


scan.onclick = function () {
    if (!isRequesting) {
        isRequesting = true
        var loadingImgObj = new loadingImg()
        loadingImgObj.show()

        let imgUrl = canvas.toDataURL("image/png");
        console.log(imgUrl)
        var data = {
            "img": imgUrl,
            "token": "THISISAFUCKINGTOKEN"
        }
        /*{#        $.getJSON($SCRIPT_ROOT + '/_add_numbers',data, function(data) {#}
        {#          $('#result').text(data.result);#}
        {#          $('input[name=a]').focus().select();#}
        {#        });#}
         */

        $.ajax({
            type: 'post',
            url: '/digit_rec',
            data: JSON.stringify(data),
            contentType: 'application/json; charset=UTF-8',
            dataType: 'json', success: function (data) {
                if (data.code == "200") {
                    var rootObj = JSON.parse(data.result)
                    showResult(rootObj.numbers, rootObj.marked_img)
                    //popPicWindow(rootObj.numbers,rootObj.marked_img)
                    //console.log(rootObj.marked_img)
                } else {
                    alert('Bad Request:'+data.code+" "+data.message)
                }
                loadingImgObj.hide()
                scan.innerHTML = "<i class=\"iconfont icon-canvas-ranlysaoma\"></i>"
                isRequesting = false
            }, error: function (xhr, type, xxx) {
                alert('Bad Request：unknown error')
                loadingImgObj.hide()
                scan.innerHTML = "<i class=\"iconfont icon-canvas-ranlysaoma\"></i>"
                isRequesting = false
            }
        });
    } else {
        alert("Loading...");
    }
}



function getColor() {
    for (let i = 0; i < aColorBtn.length; i++) {
        aColorBtn[i].onclick = function () {
            for (let i = 0; i < aColorBtn.length; i++) {
                aColorBtn[i].classList.remove("active");
                this.classList.add("active");
                activeColor = this.style.backgroundColor;
                ctx.fillStyle = activeColor;
                ctx.strokeStyle = activeColor;
            }
        }
    }
}

// let historyDeta = [];

function saveData(data) {
    (historyDeta.length === 10) && (historyDeta.shift());// 上限为储存10步，太多了怕挂掉
    historyDeta.push(data);
}

undo.onclick = function () {
    if (historyDeta.length < 1) return false;
    ctx.putImageData(historyDeta[historyDeta.length - 1], 0, 0);
    historyDeta.pop()
    var result_text = document.getElementById("info");
    result_text.innerHTML = "Result"
};

function showResult(text, imgSrc) {
    popPicWindow(text, imgSrc)
}

function loadingImg(mySetting) {
    var that = this;
    if (mySetting == "" || mySetting == undefined || typeof mySetting != "object") {
        mySetting = {};
    }
    var targetID = new Date().getTime();
    this.setting = {

        targetConater: scan,
        imgUrl: "/static/img/loading1.gif",
        imgWidth: "",
        imgClass: "",
        "targetID": targetID,
        beforeShow: function (plugin) {
        },
        afterShow: function (plugin, targetID) {
        }
    }
    this.setting = $.extend(this.setting, mySetting);
    this.getScreenWidth = function () {
        return document.documentElement.clientWidth;
    }
    this.getScreenHeight = function () {
        return document.documentElement.clientHeight;
    }
    this.show = function () {
        $("#" + that.setting.targetID).show();
    }
    this.hide = function () {
        $("#" + that.setting.targetID).hide();
    }
    this.init = function () {

        if (typeof that.setting.beforeShow == "function") {
            that.setting.beforeShow(that);
        }

        var targetHTML = '';

        if (that.setting.targetConater != "" && this.setting.targetConater != undefined) {
            targetHTML = '<img src="' + that.setting.imgUrl + '" class="' + that.setting.imgClass + '" id="' + that.setting.targetID + '" style="display:none;vertical-align: middle">';
            $(that.setting.targetConater).html(targetHTML);
        } else {
            targetHTML = '<img src="' + that.setting.imgUrl + '" class="' + that.setting.imgClass + '" style="margin: 0 auto;">';

            targetHTML = '<div id="' + that.setting.targetID + '" style="display:none;position: absolute;top:50%;left: 50%;height: ' + that.getScreenHeight() + ';width:' + that.getScreenWidth() + '">' + targetHTML + '</div>';
            $("body").append(targetHTML);
        }

        if (that.setting.imgWidth != "" && that.setting.imgWidth.indexOf("px") > 0) {
            $("#" + targetID).css("width", that.setting.imgWidth);
        }

        if (typeof that.setting.afterShow == "function") {
            that.setting.afterShow(that, targetID);
        }
    }
    this.init();
}

function popPicWindow(text, imgSrc) {
    var modal = document.getElementById('myModal');

    //var img = document.getElementById('myImg');
    var modalImg = document.getElementById("img01");
    var captionText = document.getElementById("caption");
    modal.style.display = "block";
    modalImg.src = imgSrc;
    captionText.innerHTML = "Result" + text;

    var span = document.getElementsByClassName("close")[0];

    span.onclick = function () {
        modal.style.display = "none";
    }

}

