### <a name="dnsserver">ubuntun linux 域名服务器配置</a>
vim /etc/resolvconf/resolv.conf.d/base 
nameserver 8.8.8.8 
sudo resolvconf -u 

### <a name="routeconf">ubuntu linux 路由表配置</a>
所有发往192.168.*.*通过192.168.11.1 
sudo route add -net 192.168.0.0 netmask 255.255.0.0 gw 192.168.11.1 
路由配置加到 /etc/network/interfaces 避免重启后丢失 
up sudo route add -net 172.16.0.0 netmask 255.255.0.0 gw 172.16.0.1 

### <a name="ignorenetwork">ubuntu跳过开机网络检测</a>
进入系统后修改文件/etc/systemd/system/network-online.target.wants/networking.service 
命令：sudo vim /etc/systemd/system/network-online.target.wants/networking.service 
将里面的TimeoutStartSec=5min 修改为TimeoutStartSec=2sec 


### <a name="mscoco2017">MS coco 2017 数据集</a>
http://images.cocodataset.org/zips/train2017.zip 
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip 
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip 
http://images.cocodataset.org/annotations/image_info_test2017.zip 
这些就是全部的microsoft coco数据集2017的链接了。

在linux上可以使用axel多线程工具来下载

### [2018-10-17 斗拱](./dougong)
