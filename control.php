<!DOCTYPE html>
<html>
<head>
	<script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
	<title>Gun Control</title>
</head>
<body>
	<h1>Video Stream:</h1>
	<img src="http://khaosgun.local:8080/?action=stream" alt="camera turned off or not working">
	<h2>Control</h2>
	<div id="revbutton">REV</div>
	<div id="firebutton" onmousedown="fire()" onmouseup="ceasefire()" ontouchstart="fire()" ontouchend="ceasefire()">FIRE</div>
	<div id="photobutton" onmousedown="redify()" onmouseup="greenify()" onclick="photo2()" ontouchstart="redify()" ontouchend="greenify()">PHOTO</div>
	<style>
		#firebutton {
			padding: 10px;
			margin: 10px;
		}
		#revbutton {
			padding: 10px;
			margin: 10px;
		}
		#photobutton {
			padding: 10px;
			margin: 10px;
		}
		.red {
			background-color: red;
		}
		.green {
			background-color: green;
		}
	</style>
	<script>
		$(document).ready(function(){
			$.ajax({
				url: 'pinoff.php',
				success: function(data) {
					$('.result').html(data);
				}
			});
                        $.ajax({
                                url: 'firepinoff.php',
                                success: function(data) {
                                        $('.result').html(data);
                                }
                        });
			$('#photobutton').addClass('green');
			$('#revbutton').addClass('green');
			$('#firebutton').addClass('green');
			$('#revbutton').click(function(){
				if ($('#revbutton').hasClass('green')) {
					$('#revbutton').removeClass('green');
					$('#revbutton').addClass('red');
					$.ajax({
						url: 'firepinon.php',
						success: function(data) {
							$('.result').html(data);
						}
					});
				}
				else {
					$('#revbutton').removeClass('red').addClass('green');
                                        $.ajax({
                                                url: 'firepinoff.php',
                                                success: function(data) {
                                                        $('.result').html(data);
                                                }
                                        });
				}
			});
		});
		function fire() {
			$('#firebutton').removeClass('green').addClass('red');
			$.ajax({
				url: 'pinon.php',
				success: function(data) {
					$('.result').html(data);
				}
			});
		}
		function ceasefire() {
                        $('#firebutton').removeClass('red').addClass('green');
                        $.ajax({
                                url: 'pinoff.php',
                                success: function(data) {
                                        $('.result').html(data);
                                }
                        });
                }
		function photo2() {
			$.ajax({
				url: 'takephoto.php',
				success: function(data){
					$('.result').html(data);
				}
			});
			setTimeout(function() {location.reload(); }, 1500);
		}
		function redify() {
                        $('#photobutton').removeClass('green').addClass('red');
		}
                function greenify() {
                        $('#photobutton').removeClass('red').addClass('green');
                }
		function reloadpage() {
			location.reload();
		}
	</script>
</body>
</html>
