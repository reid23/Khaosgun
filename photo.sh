sudo kill -9 `pidof mjpg_streamer`
DATE=$(date +"%Y-%m-%d_%H:%M:%S")
raspistill -o /var/www/html/photos/$DATE.jpg
/usr/local/bin/mjpg_streamer -i "input_uvc.so -r 1280x720 -d /dev/video0 -f 30 -q 80" -o "output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www"
