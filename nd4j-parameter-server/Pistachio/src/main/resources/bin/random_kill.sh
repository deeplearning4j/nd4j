#! /bin/bash
#ps ax | grep -i 'com\.yahoo\.ads\.pb\.PistachiosServer' | grep java | grep -v grep | awk '{print $1}' | xargs kill 
while true; do
	echo 'stopping server...'
	./stop_server.sh

	a=$(( $RANDOM%180 ))
	b=$(( 60+a ))
	echo 'sleeping '$a
	sleep $a
	
	echo 'starting server...'
	./start_server.sh

	a=$(( $RANDOM%180 ))
	b=$(( 60+a ))
	echo 'sleeping '$a
	sleep $a

done
	
