#! /bin/bash
set -e

git checkout master
mvn javadoc:javadoc
rm -rf /tmp/pistachio_apidocs
mkdir /tmp/pistachio_apidocs
cp -r ./target/site/apidocs /tmp/pistachio_apidocs/
git checkout gh-pages
cp -r /tmp/pistachio_apidocs/apidocs/* ./apidocs/
git commit -a -m "update apidocs"
git push origin gh-pages
git checkout master
