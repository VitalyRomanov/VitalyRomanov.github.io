---
layout: post
title: Working With Vagrant
categories: [big-data]
tags: [Vagrant]
mathjax: true
show_meta: true
published: true
---
This note will provide you some background on how to deploy virtual machines with Vagrant. More specifically, we will have a look at the basic outline and features of  `Vagrantfile` . This tutorial is a distilled version of the [official Vagrant Documentation](https://www.vagrantup.com/docs/), and most of the examples are borrowed from there. 

We will do the following:

- Spin up a generic Ubuntu VM
- Install apache server
- Perform port forwarding
- Learn how to create a multi-machine environment
- Connect multiple virtual machines with VPN

## Install Vagrant

Download and install VirtualBox from https://www.virtualbox.org/wiki/Downloads
Download and install Vagrant from http://downloads.vagrantup.com

Vagrant uses VirtualBox as a standard hypervisor. At least one hypervisor provider required to run a VM.

## Optional: Setting up Vagrant project with a local box image

Download Box images 

- [i386](https://cloud-images.ubuntu.com/vagrant/trusty/current/trusty-server-cloudimg-i386-vagrant-disk1.box)
- [amd64](https://cloud-images.ubuntu.com/vagrant/trusty/current/trusty-server-cloudimg-amd64-vagrant-disk1.box)

Use downloaded images to init your VM.

```bash
vagrant init my-box /path/to/my-box.box
```

## Setup up a Vagrant project from a repository:

Go to https://app.vagrantup.com/boxes/search and select an image that you are going to spin up. We recommend choosing**ubuntu/trusty64**. You will see that you can configure your VM using only a single command

```bash
vagrant init ubuntu/trusty64
```

![](https://i.imgur.com/ptKVCdT.png =350x)


Create a where you want to store your VM and run configuration.

## Creating a VM 

Your VM is ready, and you can now ask Vagrant to start up a VM as configured by the default Vagrantfile:

```
  vagrant up 
```

The VM is now running in Virtualbox. You can ssh into it (no password required)
as follows: 

```
  vagrant ssh # ssh into the VM
```

Further, you can log out by typing `exit` or pressing `Ctrl+D`. Even though you closed your ssh session, the VM is still working in the background. You can verify this by opening VirtualBox window.

![](https://i.imgur.com/JJIm94I.png =300x)

To terminate or suspend your VM you can type `vagrant halt` or `vagrant suspend` correspondingly. If you want to delete the instantiation of this VM configuration, type `vagrant destroy`.

## Synced Directory

Note that on the new VM, `/vagrant` is a shared directory linked with
the init directory on your host machine. On your host machine try

```bash
echo "Hello World" > hello_world.txt
```

Then log into your VM and verify the presence of the file

```bash
vagrant ssh
vagrant@vagrant-ubuntu-trusty-64:~$ cd /vagrant/
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ ls
hello_world.txt  Vagrantfile
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ cat hello_world.txt 
Hello World
vagrant@vagrant-ubuntu-trusty-64:/vagrant$ 
```

## Vagrantfile

The configuration of your VM is done with `Vagrantfile`.  When you were setting up your VM, this file was automatically added to your directory. Take a look at the content. There are several useful configs:

1. `config.vm.box` specifies the name of your VM
2. `config.vm.provider` specifies your hypervisor and additional related settings. The default hypervisor is VirtualBox
3. `config.vm.network` defines network settings for your host to see your VM
4. `config.vn.synced_folder` allows to change default shared folder path
5. `config.vm.provision` specifies additional configuration

## Synced Directory

To change the default shared directory path, simply modify the Vagrantfile

```ruby
Vagrant.configure("2") do |config|
  # other config here

  config.vm.synced_folder "src/", "/srv/website"
end
```

## Provisioning

Provision section of the Vagrantfile allows you to install additional software on your VM during its deployment. Let us create a simple example. In your init directory create `bootstrap.sh` with the following content

```bash
#!/usr/bin/env bash

apt-get update
apt-get install -y apache2
if ! [ -L /var/www ]; then
  rm -rf /var/www
  ln -fs /vagrant /var/www
fi
```

As you can see, it installs `apache2` from the repository and swaps `/vagrant` for its server directory. To execute this script during deployment, modify Vagrantfile

```ruby
Vagrant.configure("2") do |config|
  config.vm.box = "my-box"
  config.vm.provision :shell, path: "bootstrap.sh"
end
```

Provisioning happens at certain points during the lifetime of your Vagrant environment:

- On the first `vagrant up` that creates the environment, provisioning is run. If the environment was already created and the up is just resuming a machine or booting it up, they will not run unless the `--provision` flag is explicitly provided.
- When `vagrant provision` is used on a running environment.
- When `vagrant reload --provision` is called. The `--provision` flag must be present to force provisioning.

You can also bring up your environment and explicitly *not* run provisioners by specifying `--no-provision`.

## Networking

Vagrant allows to set up port forwarding to your VM. This will enable you to access a port on your machine, but actually, have all the network traffic forwarded to a specific port on the guest machine. Modify Vagrantfile according to the following

```ruby
Vagrant.configure("2") do |config|
  config.vm.box = "my-box"
  config.vm.provision :shell, path: "bootstrap.sh"
  config.vm.network :forwarded_port, guest: 80, host: 4567
end
```

You need to perform `vagrant reload` after changing networking settings. 

Since you modified the default server directory, you need to create a directory `html` in your shared folder, containing a simple html file `index.html`.

```html
Hello World!
```

Once the machine is running again, load `http://127.0.0.1:4567` in your browser. You should see a web page that is being served from the virtual machine that was automatically setup by Vagrant.

## Multi-Machine

Vagrant can define and control multiple guest machines per Vagrantfile. This is known as a "multi-machine" environment.

These machines are generally able to work together or are somehow associated with each other. Here are some use-cases people are using multi-machine environments for today:

- Accurately modeling a multi-server production topology, such as separating a web and database server.
- Modeling a distributed system and how they interact with each other.
- Testing an interface, such as an API to a service component.
- Disaster-case testing: machines dying, network partitions, slow networks, inconsistent world views, etc.

One of the simplest ways to create a multi-machine environment is 

```ruby
BOX_URL = "/path/to/image"

Vagrant.configure("2") do |config|
  config.vm.define "master" do |subconfig|
    subconfig.vm.box = "my-box1"
    subconfig.vm.box_url = BOX_URL
  end

  config.vm.define "node1" do |subconfig|
    subconfig.vm.box = "my-box2"
    subconfig.vm.box_url = BOX_URL
  end

  config.vm.define "node2" do |subconfig|
    subconfig.vm.box = "my-box3"
    subconfig.vm.box_url = BOX_URL
  end
end
```

To ssh into any of these, you will need to type ssh command with a name

```bash
vagrant ssh my-box1
```

To destroy all VMs, type 

```bash
vagrant destroy -f
```

## Multi-Machine Networking

To allow machines to communicate with each other, specify additional networking parameters. First, each VM needs a unique hostname. By default, each of the VMs has the same hostname (`vagrant`). Change this with

```ruby
subconfig.vm.hostname = "a.host.name"
```

Next, we need a way of getting the IP address for a hostname. For this, we’ll use DNS – or mDNS to be more precise. 

On Ubuntu, mDNS is provided by Avahi. To install Avahi on each node, we’ll use Vagrant’s [provisioning feature](https://www.vagrantup.com/docs/provisioning/). 

Before the last `end` in the `Vagrantfile`, we’ll add this code block:

```ruby
config.vm.provision "shell", inline: <<-SHELL
  apt-get install -y avahi-daemon libnss-mdns
SHELL
```

This will call `apt-get install -y avahi-daemon libnss-mdns` on every VM.

Last, we need to connect the VMs through a [private network](https://www.vagrantup.com/docs/networking/private_network.html). 

For each VM, we need to add a config like this (where each VM will have a different IP address):

```ruby
subconfig.vm.network :private_network, ip: "10.0.0.10"
```

You can now call `vagrant up` and then ssh into any of the VMs:

```bash
vagrant ssh my-box1
```

From there you can ping any other VM by using their hostname (plus `.local` at the end):

```bash
ping VM_addr.local
```

## Multiple Provisioners

Multiple `config.vm.provision` methods can be used to define multiple provisioners. These provisioners will be run in the order they're defined. This is useful for a variety of reasons, but most commonly it is used so that a shell script can bootstrap some of the systems so that another provisioner can take over later.

## Additional Features

For the information about additional features visit documentation web site

- [Push](https://www.vagrantup.com/docs/push/)
- [Plugins](https://www.vagrantup.com/docs/plugins/)
- [Providers](https://www.vagrantup.com/docs/providers/)
- [Triggers](https://www.vagrantup.com/docs/triggers/)
- [Other](https://www.vagrantup.com/docs/other/)

# Troubleshooting
- If `Vagrant` does not see `VirtualBox` even though you have installed it, try downgrading to `VirtualBox 5.1`

*[hypervisor]: A hypervisor or virtual machine monitor (VMM) is computer software, firmware or hardware that creates and runs virtual machines. A computer on which a hypervisor runs one or more virtual machines is called a host machine, and each virtual machine is called a guest machine.