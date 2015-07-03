

service Pistachios {
   binary lookup(1: i64 id),
   bool store(1:i64 id, 2:binary value),
   bool processEvent(1:binary event, 2:string jar_url, 3:string class_full_name),
   bool registerPartitioner(1:string jar_url, 2:string class_full_name),
}

