#! /bin/bash
base_dir=$(readlink -f $(dirname $0))
java -cp $base_dir/../lib/*:$base_dir/../config/ -Djava.library.path=$base_dir/../lib -Dlog4j.configuration=file://$base_dir/../config/commandslog4j.xml -DconfigPath=$base_dir/../config org.nd4j.paramserver.pistachio.pb.PistachiosFormatter $*
