#! /bin/bash
ps ax | grep -i 'com\.yahoo\.ads\.pb\.PistachiosServer' | grep java | grep -v grep | awk '{print $1}' | xargs kill 
