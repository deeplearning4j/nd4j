Pistachio
==========
[![CI](https://travis-ci.org/lyogavin/Pistachio.svg)](https://travis-ci.org/lyogavin/Pistachio)
### Low latency, Strong consistency, Fault tolerant

Pistachio is a distributed key value store system. Data can be replicated with n replicas with strong consistency gurantees. Up to (n-1) failures can be tolerated.

### Proven in massive scale production 

Pistachio is being used as the user profile storage for massive scale ad serving products in Yahoo. 10+ billions of user profiles are being stored with ~2 million reads QPS and ~0.5 million writes QPS. It guarantees strong consistency and fault-tolerance. We have hundres of servers in 8 data centers all over the globe.

### Re-invent the storage system for big data cloud compute

Data locality is the key to the performance of cloud computing. Run time can have 100x difference to process the same data with different data locality. HDFS is a very reliable storage system and mostly commonly used in large scale cloud computing. But it's too slow already. We've successfully used Pistachio to achieve 100x faster cloud compute. Storage can no longer be a external system to the compute. Compute can be embedded to the storage. Best compute performacne can be achieved this way.

### Flexible interfaces to embed compute logic to data storage
Traditional pattern to access storage system is to access from a client lib, you read the data, do some compute, then write it back. There can be lots of round trip overhead. Pistachio supports lots of flexible interfaces for the client to embed callbacks or processing logics to the storage server side to avoid the round trips.

### Flexible local storage engine plugin
Different use case may need different durability. In some cases the storage is only used for short term compute, then it'll be persisted to HDFS. Then you want to have memory only storage with replications only for fault tolerance. In some cases, when you have many random read/write, and using SSD as the local storage may give you the best performacne. Pistachio supports flexible local storage engines in-mem, Kyoto Cabinet(best for SSD) and Rocks DB.


http://lyogavin.github.io/Pistachio/

Copyright 2014 Yahoo! Inc. Licensed under the Apache License 2.0

