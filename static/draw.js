//for canvas drawing used code from here: https://github.com/zealerww/digits_recognition/blob/master/digits_recognition/static/draw.js
var drawing = false;

var context;

var offset_left = 0;
var offset_top = 0;

//function startup() {
//  var el = document.body;
//  el.addEventListener("touchstart", handleStart, false);
//  el.addEventListener("touchend", handleEnd, false);
//  el.addEventListener("touchcancel", handleCancel, false);
//  el.addEventListener("touchleave", handleEnd, false);
//  el.addEventListener("touchmove", handleMove, false);
//}


//canvas functions
function start_canvas () {
    var canvas = document.getElementById("the_stage");
    context = canvas.getContext("2d");
    canvas.onmousedown = function (event) {mousedown(event)};
    canvas.onmousemove = function (event) {mousemove(event)};
    canvas.onmouseup = function (event) {mouseup(event)};

    //canvas.onmouseout = function (event) {mouseup(event)};
    //canvas.ontouchstart = function (event) {touchstart(event)};
    //canvas.ontouchmove = function (event) {touchmove(event)};
    //canvas.ontouchend = function (event) {touchend(event)};
    for (var o = canvas; o ; o = o.offsetParent) {
    offset_left += (o.offsetLeft - o.scrollLeft);
    offset_top  += (o.offsetTop - o.scrollTop);
    }

     draw();
  //var el =document.body;
  //el.addEventListener("touchstart", handleStart, false);
  //el.addEventListener("touchend", handleEnd, false);
  //el.addEventListener("touchcancel", handleCancel, false);
  //el.addEventListener("touchleave", handleEnd, false);
  //el.addEventListener("touchmove", handleMove, false);
  //  draw();
}

function getPosition(evt) {
    evt = (evt) ?  evt : ((event) ? event : null);
    var left = 0;
    var top = 0;
    var canvas = document.getElementById("the_stage");

    if (evt.pageX) {
    left = evt.pageX;
    top  = evt.pageY;
    } else if (document.documentElement.scrollLeft) {
    left = evt.clientX + document.documentElement.scrollLeft;
    top  = evt.clientY + document.documentElement.scrollTop;
    } else  {
    left = evt.clientX + document.body.scrollLeft;
    top  = evt.clientY + document.body.scrollTop;
    }
    left -= offset_left;
    top -= offset_top;

    return {x : left, y : top}; 
}


function mousedown(event) {
    drawing = true;
    var location = getPosition(event);
    context.lineWidth = 8.0;
    context.strokeStyle = "#000000";
    context.beginPath();
    context.moveTo(location.x, location.y);
}


function mousemove(event) {
    if (!drawing) 
        return;
    var location = getPosition(event);
    context.lineTo(location.x, location.y);
    context.stroke();
}


function mouseup(event) {
    if (!drawing) 
        return;
    mousemove(event);
    context.closePath();
    drawing = false;
}


function draw() {
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 200, 200);
}

function clearCanvas() {
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 200, 200)

    document.getElementById("hidable").style.display = "none";


}


function predict_result() {

    var canvas = document.getElementById("the_stage");
    var dataURL = canvas.toDataURL('image/jpg');

    $.ajax({
        type: "POST",
        url: "/hook2",
        data:{
			imageBase64: dataURL

		}
    }).done(function(resp_data) {

        var response = JSON.parse(resp_data)

        if (response  == "Can't recognize - nothing is drawn!") {
            alert(response)
        } else {

        document.getElementById("hidable").style.display = "block";

        document.getElementById("fnn1").innerHTML = response['fnn1'];
		document.getElementById("fnn2").innerHTML = response['fnn2'];
		document.getElementById("fnn3").innerHTML = response['fnn3'];

		document.getElementById("fnn1_proba").innerHTML = response['fnn1_proba'];
		document.getElementById("fnn2_proba").innerHTML = response['fnn2_proba'];
		document.getElementById("fnn3_proba").innerHTML = response['fnn3_proba'];

		document.getElementById("cnn1").innerHTML = response['cnn1'];
		document.getElementById("cnn2").innerHTML = response['cnn2'];
		document.getElementById("cnn3").innerHTML = response['cnn3'];

		document.getElementById("cnn1_proba").innerHTML = response['cnn1_proba'];
		document.getElementById("cnn2_proba").innerHTML = response['cnn2_proba'];
		document.getElementById("cnn3_proba").innerHTML = response['cnn3_proba'];

        }


    });
}


onload = start_canvas;