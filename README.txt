
HeavyCUDA release Mar 18th 2014 - Initial Release
-------------------------------------------------------------

***************************************************************
If you find this tool useful and like to support its continued 
          development, then consider a donation.

   LTC donation address: LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
   BTC donation address: 16hJF5mceSojnTD3ZTUDqdRhDyPJzoRakM
   YAC donation address: Y87sptDEcpLkLeAuex6qZioDbvy1qXZEj4
   VTC donation address: VrjeFzMgvteCGarLw85KivBzmsiH9fqp4a
   MAX donation address: mHrhQP9EFArechWxTFJ97s9D3jvcCvEEnt
  DOGE donation address: DT9ghsGmez6ojVdEZgvaZbT2Z3TruXG6yP
 PANDA donation address: PvgtxJ2ZKaudRogCXfUMLXVaWUMcKQgRed
   MRC donation address: 1Lxc4JPDpQRJB8BN4YwhmSQ3Rcu8gjj2Kd
   HVC donation address: HNN3PyyTMkDo4RkEjkWSGMwqia1yD8mwJN
***************************************************************

>>> Introduction <<<

This is a CUDA accelerated mining application for use with
HeavyCoin and FugueCoin. We did not take effort on usability,
so please set your parameters carefuly.

THIS PROGRAMM IS PROVIDED "AS-IS", USE IT AT YOUR OWN RISK!

If you're interessted and read the source-code, please excuse
that the most of our comments are in german.

>>> Command Line Interface <<<

This code is based on the pooler cpuminer 2.3.2 release and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use
                          heavy       use to mine Heavycoin
                          fugue256    use to mine Fuguecoin

  -o, --url=URL         URL of mining server (default: " DEF_RPC_URL ")
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
  -v, --vote		Heavycoin block vote (default: 512)
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs in your system)
  -r, --retries=N       number of times to retry if a network call fails
                          (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 15)
  -T, --timeout=N       network timeout, in seconds (default: 270)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 5)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
  -q, --quiet           disable per-thread hashmeter output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
  -B, --background      run the miner in the background
      --benchmark       run in offline benchmark mode
  -c, --config=FILE     load a JSON-format configuration file
  -V, --version         display version information and exit
  -h, --help            display this help text and exit

>>> Examples <<<

Example for Heavycoin Mining on heavycoinpool.com with a single gpu in your system

cudaminer.exe -t 1 -a heavy -o stratum+tcp://stratum01.heavycoinpool.com:5333 -u <<username.worker>> -p <<workerpassword>> -v 512



Example for Heavycoin Mining on hvc.1gh.com with a dual gpu in your system

cudaminer.exe -t 2 -a heavy -o stratum+tcp://hvcpool.1gh.com:5333 -u <<WALLET>> -p x -v 512



Example for Fuguecoin solo-mining with 4 gpu's in your system and a Fuguecoin-wallet running on localhost

cudaminer.exe -q -s 1 -t 4 -a fugue256 -o http://localhost:9089 -u <<myusername>> -p <<mypassword>>

For solo-mining you typically use -o 127.0.0.1:xxxx where xxxx represents
the RPC portnumber specified in your wallet's .conf file and you have to
pass the same username and password with -O as specified in the wallet's
.conf file. The wallet must also be started with the -server option and
the server flag in the wallet's .conf file set to 1


>>> Additional Notes <<<

This code should be running on nVidia GPUs ranging from compute capability
2.0 up to compute capability 3.5. Just don't expect any hashing miracles
from your old clunkers.

>>> RELEASE HISTORY <<<

  March, 18 2014 initial release.


>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner, Christian H. (Germany): CUDA implementation 

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler), it's original HVC-fork
and the HVC-fork available at hvc.1gh.com

Source code is included to satisfy GNU GPL V2 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
   Christian H. ( Chris84 )
