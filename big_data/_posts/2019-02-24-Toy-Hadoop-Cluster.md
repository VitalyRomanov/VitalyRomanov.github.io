---
layout: post
title: How to Up Hadoop Cluster
categories: [big-data]
tags: [Hadoop]
mathjax: true
show_meta: true
published: true
---
# Hadoop: Distributed Cluster, YARN, MapReduce

This document is a tutorial on how to set up a distributed Hadoop cluster. It contains steps necessary for minimal working configuration for the cluster in the simplest scenario - launching an example program. Remember, Hadoop applies the parameters specified in configuration only when those parameters are used. Consequently, parameters that work for one task will not necessarily work for another, make sure to recheck all parameters for new tasks. In the case of incorrect parameters, the execution of a Hadoop program will be terminated and the error details will be written to a log file. Consult with the log file to identify the error cause. There is a troubleshooting section at the end of this document, where you can find solutions for most frequent problems. 

##### VM Info
Credentials used in virtual machine image
**username:** vagrant
**password:** vagrant

## Prerequisites
To complete this tutorial you need to install Vagrant virtualization software. VirtualBox is recommended as Vagrant's hypervisor (no other configuration was tested). To expedite the system set up, download [~~virtual machine image~~](https://instant.io/#9feb69968a2dbdf5346c8d87b27b62cc1c9763ff) (1.2Gb) [[Mirror 1](https://1drv.ms/u/s!AlssRXCRRXhH2U3VqXuUp1RqF6f0), [Mirror 2](https://drive.google.com/open?id=17xKtmA39ncKmRXh4GskPao1VBbY0bB99)].


Create `Vagrantfile` with the following content (for mode information about `Vagrantfile` read previous post)
```ruby
# -*- mode: ruby -*-
# vi: set ft=ruby :
BOX_PATH = 'hadoop_image.box'

Vagrant.configure("2") do |config|

 config.vm.define "server-1" do |subconfig|
   subconfig.vm.box = "server-1" #BOX_IMAGE
   subconfig.vm.box_url = BOX_PATH
   subconfig.vm.hostname = "server-1"
   subconfig.vm.network :private_network, ip: "10.0.0.11"
   subconfig.vm.network "forwarded_port", guest: 8088, host: 8088
   subconfig.vm.provider "virtualbox" do |v|
    v.memory = 512
   end
 end

 config.vm.define "server-2" do |subconfig|
   subconfig.vm.box = "server-2" #BOX_IMAGE
   subconfig.vm.box_url = BOX_PATH
   subconfig.vm.hostname = "server-2"
   subconfig.vm.network :private_network, ip: "10.0.0.12"
   subconfig.vm.provider "virtualbox" do |v|
    v.memory = 512
   end
 end

 config.vm.define "server-3" do |subconfig|
   subconfig.vm.box = "server-3" #BOX_IMAGE
   subconfig.vm.box_url = BOX_PATH
   subconfig.vm.hostname = "server-3"
   subconfig.vm.network :private_network, ip: "10.0.0.13"
   subconfig.vm.provider "virtualbox" do |v|
    v.memory = 512
   end
 end

end
```

Place `hadoop_image.box` to the same folder as `Vagrantfile` and start VMs.

### System requirements
Current configuration creates 3 virtual machines, each with 512Mb of RAM. You can decrease it to 512Mb, however, some examples may fail to execute.

Each virtual machine occupies ~4Gb right after start. The size of a virtual machine can increase after you start working with it. It is recommended to have at least 20Gb of disk space.

### Java 8
We are going to use Java 8. Java 9 is also compatible with Hadoop, but there are some issues. Moreover, we are going to configure Hadoop to work with Spark, and there are also some compatibility issues between Scala and Java versions above 8.

## Prepare the Cluster

Start the cluster by executing `vagrant up`. 

**WARNIGN:** *Current configuration will occupy 15-20Gb  of your disk space. If you have less than 4Gb of memory, it is highly recommended to find another machine before proceeding with this tutorial.*

Open `Vagrantfile` and scan through the configuration. You have 3 VMs configured: `server-1`, `server-2`, `server-3`. Each one is assigned a dedicated IP address. Remember that you can login into a VM using `vagrant ssh vmname`, i.e.
```bash
vagrant ssh server-1
vagrant ssh server-2
vagrant ssh server-3
```

### Configuring Host Names

For proper operation, Hadoop requires all the nodes to locate each other using domain names.  Specifically, Hadoop prefers the configuration to be performed with domain names rather than IP addresses. In order for our system to resolve domain names, we need a Domain Name Service (DNS). The cheapest way to set up a resemblance of DNS is [/etc/hosts](http://man7.org/linux/man-pages/man5/hosts.5.html). In this file, you specify a mapping between domain names and their IPs. Whenever you make a request to a domain, the system first checks `/etc/hosts` and only then tries to resolve with other DNS services. Now, we want to enforce the following mapping 
```
10.0.0.11 server-1
10.0.0.12 server-2
10.0.0.13 server-3
```

**NOTE:** *Current If you run your cluster elsewhere, you can adopt the any other domain name convention. Then adjust all the following configs accordingly.*

Set `/etc/hosts` on all nodes
```
127.0.0.1	localhost

# The following lines are desirable for IPv6 capable hosts
::1     localhost ip6-localhost ip6-loopback
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters

10.0.0.11 server-1
10.0.0.12 server-2
10.0.0.13 server-3
```


### Distributing Credentials

For proper functioning of HDFS and YARN, namenode and resource manager need ssh access to datanodes and nodemanagers. In this tutorial, we are adopting a simplified architecture of a distributed system, where the same machine plays the roles of HDFS namenode and YARN resource manager. In reality, this setting is impractical but works now for educational purposes. 

We are going to use `server-1` for namenode and resource manager, this node will need ssh access to all other nodes. For this reason, access keys should be distributed across nodes.

On `server-1`, generate a key with (in case you don't have the key yet)
```bash
ssh-keygen -b 4096 
```
and distribute this key to every node in the cluster with 
```bash
ssh-copy-id -i $HOME/.ssh/id_rsa.pub vagrant@node-address
```
Copy the key to all nodes, including `server-1`. Check whether you have succeeded with 
```bash
ssh server-1
ssh server-2
ssh server-3
```
You should be able to ssh without password.

### Configure Environment Variables

When starting Hadoop services, master node needs to start necessary daemons on remote nodes. This is done by means of ssh. Other than the access, master node should be able to execute Hadoop binaries. To help locate those binaries, modify `PATH` environment variable

```bash
echo "PATH=/home/vagrant/hadoop/bin:/home/vagrant/hadoop/sbin:$PATH" >> ~/.bashrc
```

The effect of the command above will take place after the next login. Make sure you logout from `server-1` at least once to proceed with the instruction.

## Configure Hadoop

Every VM contains Hadoop binaries located in `/home/vagrant/hadoop`. Configuration files reside in `~/hadoop/etc/hadoop`. 


**INFO:** *We are going to configure `server-1` first and then copy the configuration to other nodes. Assume further commands are executed on `server-1` unless specified otherwise.*

### Default Java Environment

Change variable `JAVA_HOME` in `~/hadoop/etc/hadoop/hadoop-env.sh` to `/usr/lib/jvm/java-1.8.0-openjdk-amd64` (because Java already installed in VM image) or other path where you have your Java binaries installed. 

### HDFS

#### Hadoop Temp

Change the current directory and create a folder for temporary files (if this folder is not specified you may experience problems after restarting VMs).
```bash
cd ~
mkdir hadoop_tmp
```
Configure [`hdfs-site.xml`](https://hadoop.apache.org/docs/r3.1.1/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml):

```xml
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/home/vagrant/hadoop_tmp</value>
    </property>
</configuration>
```


#### Namenode

Configure the `core-site.xml` and specify the namenode address

```xml
<configuration>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://server-1:9000</value>
    </property>
</configuration>
```

#### Configure Workers

Edit the file `workers`

```
server-1
server-2
server-3
```

#### Distribute Configuration

To copy the configuration to other nodes we will use `scp` command. 

```bash
for node in another-server-address-1 another-server-address-2; do
    scp conf/path/on/local/machine/* $node:conf/path/on/remote/machine/;
done
```
**WARNING:** *Now, you need to create `hadoop_tmp` directories on all nodes. *

#### Format HDFS

The first time you start up HDFS, it must be formatted. Continue working on the `server-1` and format a new distributed filesystem:
```bash
hdfs namenode -format
```

#### Start HDFS 

All of the HDFS processes can be started with a utility script. On the `server-1` execute:
```bash
start-dfs.sh
```
`server-1` will connect to the rest of the nodes and start corresponding services.

#### Check Correct HDFS Functioning

```bash
hdfs dfsadmin -report
```
You should be able to run this command on any of hadoop cluster nodes and expect the following output


```
>> hdfs dfsadmin -report

Configured Capacity: 93497118720 (87.08 GB)
Present Capacity: 78340636672 (72.96 GB)
DFS Remaining: 78340562944 (72.96 GB)
DFS Used: 73728 (72 KB)
DFS Used%: 0.00%
Replicated Blocks:
	Under replicated blocks: 0
	Blocks with corrupt replicas: 0
	Missing blocks: 0
	Missing blocks (with replication factor 1): 0
	Pending deletion blocks: 0
Erasure Coded Block Groups: 
	Low redundancy block groups: 0
	Block groups with corrupt internal blocks: 0
	Missing block groups: 0
	Pending deletion blocks: 0

-------------------------------------------------
Live datanodes (3):

Name: 10.0.0.11:9866 (server-1)
Hostname: server-1
Decommission Status : Normal
Configured Capacity: 31165706240 (29.03 GB)
DFS Used: 24576 (24 KB)
Non DFS Used: 3447148544 (3.21 GB)
DFS Remaining: 26111803392 (24.32 GB)
DFS Used%: 0.00%
DFS Remaining%: 83.78%
Configured Cache Capacity: 0 (0 B)
Cache Used: 0 (0 B)
Cache Remaining: 0 (0 B)
Cache Used%: 100.00%
Cache Remaining%: 0.00%
Xceivers: 1
Last contact: Wed Feb 13 09:38:47 PST 2019
Last Block Report: Wed Feb 13 09:38:29 PST 2019
Num of Blocks: 0


Name: 10.0.0.12:9866 (server-2)
Hostname: server-2
Decommission Status : Normal
Configured Capacity: 31165706240 (29.03 GB)
DFS Used: 24576 (24 KB)
Non DFS Used: 3444572160 (3.21 GB)
DFS Remaining: 26114379776 (24.32 GB)
DFS Used%: 0.00%
DFS Remaining%: 83.79%
Configured Cache Capacity: 0 (0 B)
Cache Used: 0 (0 B)
Cache Remaining: 0 (0 B)
Cache Used%: 100.00%
Cache Remaining%: 0.00%
Xceivers: 1
Last contact: Wed Feb 13 09:38:46 PST 2019
Last Block Report: Wed Feb 13 09:38:28 PST 2019
Num of Blocks: 0


Name: 10.0.0.13:9866 (server-3)
Hostname: server-3
Decommission Status : Normal
Configured Capacity: 31165706240 (29.03 GB)
DFS Used: 24576 (24 KB)
Non DFS Used: 3444572160 (3.21 GB)
DFS Remaining: 26114379776 (24.32 GB)
DFS Used%: 0.00%
DFS Remaining%: 83.79%
Configured Cache Capacity: 0 (0 B)
Cache Used: 0 (0 B)
Cache Remaining: 0 (0 B)
Cache Used%: 100.00%
Cache Remaining%: 0.00%
Xceivers: 1
Last contact: Wed Feb 13 09:38:46 PST 2019
Last Block Report: Wed Feb 13 09:38:28 PST 2019
Num of Blocks: 0
```

If the command above fails or the number of nodes is less then three, refer to the troubleshooting section for possible solutions.


#### Working with File System

Download [Alice in Wonderland](https://www.dropbox.com/s/vvq7l7q0umt3kwn/alice.txt?dl=1). Copy the text file to one of the nodes (use Vagrant's shared folders).

Place the text file onto hdfs using `hdfs put`.
```bash
hdfs dfs -put path/on/locl/machine path/on/hdfs
```

HDFS copies to user directory by default, but it does not exist. 
```bash
hdfs dfs -mkdir /user/
hdfs dfs -mkdir /user/vagrant
```

After you copied the file, you should be able to see it on HDFS
```bash
hdfs dfs -ls
```
Most of unix filesystem commands are available
```bash
hdfs dfs -rm alice.txt
```

### YARN

Resource manager YARN usually runs on a dedicated machine. Since many of us are limited with resources, we place it on the same node as namenode.

The file responsible for YARN configuration is `yarn-site.xml`. More detailed information about the available options can be found in the [official documentation](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/ClusterSetup.html).

#### Resource Manager
```xml
<configuration>
    <property>
            <name>yarn.resourcemanager.hostname</name>
            <value>server-1</value>
    </property>
</configuration>
```

As you can see, we are going to run the resource manager on the same machine as the namenode. 


#### MapReduce

MapReduce has its own configuration file `mapred-site.xml`. Configure MapReduce framework
```xml
<configuration>
    <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
    </property>
</configuration>
```

#### Start YARN

Distribute the configuration among nodes and start YARN daemons
```bash
start-yarn.sh
```

Check the status with 
```bash
>> yarn node -list
INFO client.RMProxy: Connecting to ResourceManager at /10.0.0.11:8032
Total Nodes:3
         Node-Id	     Node-State	Node-Http-Address	Number-of-Running-Containers
  server-1:35921	        RUNNING	    server-1:8042	                           0
server-2:34747	        RUNNING	  server-2:8042	                           0
server-3:40715	        RUNNING	  server-3:8042	                           0

```

#### Finish configuring YARN

Even though you have successfully started YARN services, that does not mean that everything is going to work. We give you a shortcut to the working solution. In `mapred-site.xml` configure the absolute path to Hadoop installation. This is needed for the worker to find neccessary classes successfully.
```xml
<configuration>
    <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
    </property>
    <property>
	    <name>yarn.app.mapreduce.am.env</name>
  	    <value>HADOOP_MAPRED_HOME=/home/vagrant/hadoop</value>
    </property>
    <property>
  	    <name>mapreduce.map.env</name>
  	    <value>HADOOP_MAPRED_HOME=/home/vagrant/hadoop</value>
    </property>
    <property>
  	    <name>mapreduce.reduce.env</name>
    	    <value>HADOOP_MAPRED_HOME=/home/vagrant/hadoop</value>
    </property>
    <property>
            <name>yarn.app.mapreduce.am.resource.mb</name>
            <value>768</value>
    </property>
    <property>
            <name>mapreduce.map.memory.mb</name>
            <value>728</value>
    </property>
    <property>
            <name>mapreduce.reduce.memory.mb</name>
            <value>728</value>
    </property>
</configuration>
```
Apache recommends allocating 8GB of RAM to a Hadoop node. Since we are running all three nodes on the same machine, such a wealth of resources is doubtful. YARN performs different checks for memory allocation, even if it does not actually need the resource yet. Configure `yarn-site.xml`
```xml
<configuration>
    <property>
            <name>yarn.resourcemanager.hostname</name>
            <value>server-1</value>
    </property>
    <property>
	    <name>yarn.nodemanager.aux-services</name>
	    <value>mapreduce_shuffle</value>
    </property>
    <property>
            <name>yarn.nodemanager.resource.memory-mb</name>
            <value>768</value>
    </property>
    <property>
            <name>yarn.scheduler.maximum-allocation-mb</name>
            <value>768</value>
    </property>

    <property>
            <name>yarn.scheduler.minimum-allocation-mb</name>
            <value>256</value>
    </property>
    <property>
            <name>yarn.nodemanager.vmem-pmem-ratio</name>
            <value>5.1</value>
    </property>
</configuration>
```
Distribute this configuration to all nodes, then start YARN again. 

#### Run MapReduce Job

```bash
hadoop jar hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.1.1.jar pi 20 10
```

You can expect to see similar output 
```
2019-02-13 10:12:46,877 INFO mapreduce.Job: Job job_1550081099154_0001 completed successfully
2019-02-13 10:12:47,038 INFO mapreduce.Job: Counters: 53
	File System Counters
		FILE: Number of bytes read=2206
		FILE: Number of bytes written=21714789
		FILE: Number of read operations=0
		FILE: Number of large read operations=0
		FILE: Number of write operations=0
		HDFS: Number of bytes read=26590
		HDFS: Number of bytes written=215
		HDFS: Number of read operations=405
		HDFS: Number of large read operations=0
		HDFS: Number of write operations=3
	Job Counters 
		Launched map tasks=100
		Launched reduce tasks=1
		Data-local map tasks=100
		Total time spent by all maps in occupied slots (ms)=3272064
		Total time spent by all reduces in occupied slots (ms)=826509
		Total time spent by all map tasks (ms)=1090688
		Total time spent by all reduce tasks (ms)=275503
		Total vcore-milliseconds taken by all map tasks=1090688
		Total vcore-milliseconds taken by all reduce tasks=275503
		Total megabyte-milliseconds taken by all map tasks=837648384
		Total megabyte-milliseconds taken by all reduce tasks=211586304
	Map-Reduce Framework
		Map input records=100
		Map output records=200
		Map output bytes=1800
		Map output materialized bytes=2800
		Input split bytes=14790
		Combine input records=0
		Combine output records=0
		Reduce input groups=2
		Reduce shuffle bytes=2800
		Reduce input records=200
		Reduce output records=0
		Spilled Records=400
		Shuffled Maps =100
		Failed Shuffles=0
		Merged Map outputs=100
		GC time elapsed (ms)=28868
		CPU time spent (ms)=131670
		Physical memory (bytes) snapshot=21592756224
		Virtual memory (bytes) snapshot=237003071488
		Total committed heap usage (bytes)=12206575616
		Peak Map Physical memory (bytes)=221532160
		Peak Map Virtual memory (bytes)=2352791552
		Peak Reduce Physical memory (bytes)=128483328
		Peak Reduce Virtual memory (bytes)=2351853568
	Shuffle Errors
		BAD_ID=0
		CONNECTION=0
		IO_ERROR=0
		WRONG_LENGTH=0
		WRONG_MAP=0
		WRONG_REDUCE=0
	File Input Format Counters 
		Bytes Read=11800
	File Output Format Counters 
		Bytes Written=97
Job Finished in 393.5 seconds
Estimated value of Pi is 3.14800000000000000000
```

If you could not get the same output, try to resolve issue by reading logs.

#### Second MapReduce Job

Then, run `wordcount` example

```bash
hadoop jar hadoop⁩/⁨share⁩/⁨hadoop⁩/⁨mapreduce⁩/hadoop-mapreduce-examples-3.1.1.jar wordcount TextFile OutputDirectory
```
Both `TextFile` and `OtputDirectory` are paths in HDFS. Use `alice.txt` as`TextFile`.
If no error has ocurred, ouput directory will contain two files
```
hdfs dfs -ls OtputDirectory
Found 2 items
-rw-r--r--   3 vagrant supergroup          0 2018-10-02 08:47 OtputDirectory/_SUCCESS
-rw-r--r--   3 vagrant supergroup     557120 2018-10-02 08:47 OtputDirectory/part-r-00000
```

You can examine the content of the output
```bash
hdfs dfs -tail OtputDirectory/part-r-00000
```


# Troubleshooting

## General Troubleshooting with Logs

The default location of log files is `~/hadoop/logs`. There you can find logs for namenode, datanode, resourcemanager, and nodemanager depending on what is running on a particular machine.

## Formatting HDFS
When you decide to format the file system, you need to clean data on both namenode and datanodes. 
1. Stop the filesystem with `stop-dfs.sh`.
2. Log into each datanode and remove the data `rm -r ~/hadoop_tmp/*`
3. Format the filesystem `hdfs namenode -format`
4. Now you can restart the filesystem `start-dfs.sh`

## HDFS
### hdfs dfsadmin reports less than three nodes
Make sure
- You distributed `ssh` key
- You added workers to `etc/hadoop/workers`
- You distributed configuration
- You formatted HDFS

### Cannot connect to 10.0.0.11:9000
- Namenode did not start

## YARN
### Yarn nodes do not start
- Check you copied the configuration
- Check your /etc/hosts is set up properly

## Other known errors
- `Caused by: java.net.UnknownHostException: datanode-2` - incorrect hostnames
- namenode log: 
```
2018-12-26 11:28:34,266 ERROR org.apache.hadoop.hdfs.server.namenode.NameNode: Failed to start namenode.
org.apache.hadoop.hdfs.server.common.InconsistentFSStateException: Directory /private/tmp/hadoop-LTV/dfs/name is in an inconsistent state: storage directory does not exist or is not accessible.
	at org.apache.hadoop.hdfs.server.namenode.FSImage.recoverStorageDirs(FSImage.java:376)
	at org.apache.hadoop.hdfs.server.namenode.FSImage.recoverTransitionRead(FSImage.java:227)
	at org.apache.hadoop.hdfs.server.namenode.FSNamesystem.loadFSImage(FSNamesystem.java:1086)
	at org.apache.hadoop.hdfs.server.namenode.FSNamesystem.loadFromDisk(FSNamesystem.java:714)
	at org.apache.hadoop.hdfs.server.namenode.NameNode.loadNamesystem(NameNode.java:632)
	at org.apache.hadoop.hdfs.server.namenode.NameNode.initialize(NameNode.java:694)
	at org.apache.hadoop.hdfs.server.namenode.NameNode.<init>(NameNode.java:937)
	at org.apache.hadoop.hdfs.server.namenode.NameNode.<init>(NameNode.java:910)
	at org.apache.hadoop.hdfs.server.namenode.NameNode.createNameNode(NameNode.java:1643)
	at org.apache.hadoop.hdfs.server.namenode.NameNode.main(NameNode.java:1710)
2018-12-26 11:28:34,269 INFO org.apache.hadoop.util.ExitUtil: Exiting with status 1: org.apache.hadoop.hdfs.server.common.InconsistentFSStateException: Directory /private/tmp/hadoop-LTV/dfs/name is in an inconsistent state: storage directory does not exist or is not accessible.
2018-12-26 11:28:34,277 INFO org.apache.hadoop.hdfs.server.namenode.NameNode: SHUTDOWN_MSG: 
```
Need to reformat HDFS
- datanode log
```
2018-12-26 11:28:58,118 INFO org.apache.hadoop.ipc.Client: Retrying connect to server: 10.240.16.166:9000. Already tried 5 time(s); retry policy is RetryUpToMaximumCountWithFixedSleep(maxRetries=10, sleepTime=1000 MILLISECONDS)
```
Namenode is down or domain name is configured incorrectly


# References
1. [Creating Vagrant Images](https://scotch.io/tutorials/how-to-create-a-vagrant-base-box-from-an-existing-one)
2. [Configuring Network Names](https://www.cloudera.com/documentation/enterprise/5-15-x/topics/cdh_ig_networknames_configure.html)

*[hypervisor]: A hypervisor or virtual machine monitor (VMM) is computer software, firmware or hardware that creates and runs virtual machines. A computer on which a hypervisor runs one or more virtual machines is called a host machine, and each virtual machine is called a guest machine.