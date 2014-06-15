
ccMiner release 1.2 (June 15th 2014) - "Killer Groestl + X13"
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
   HVC donation address: HNN3PyyTMkDo4RkEjkWSGMwqia1yD8mwJN
   GRS donation address: FmJKJAhvyHWPeEVeLQHefr2naqgWc9ABTM
   MYR donation address: MNHM7Q7HVfGpKDJgVJrY8ofwvmeugNewyf
   JPC donation address: JYFBypVDkk583yKWY4M46TG5vXG8hfgD2U
   SFR donation address: SR4b87aEnPfTs77bo9NnnaV21fiF6jQpAp
   MNC donation address: MShgNUSYwybEbXLvJUtdNg1a7rUeiNgooK
   BTQ donation address: 13GFwLiZL2DaA9XeE733PNrQX5QYLFsonS
***************************************************************

>>> Introduction <<<

This is a CUDA accelerated mining application for use with

HeavyCoin & MjollnirCoin
FugueCoin
GroestlCoin & Myriad-Groestl
JackpotCoin
QuarkCoin family & AnimeCoin
TalkCoin
DarkCoin and other X11 coins

where some of these coins have a VERY NOTABLE nVidia advantage
over competing AMD (OpenCL) implementations.

We did not take a big effort on improving usability, so please set
your parameters carefuly.

THIS PROGRAMM IS PROVIDED "AS-IS", USE IT AT YOUR OWN RISK!

If you're interessted and read the source-code, please excuse
that the most of our comments are in german.

>>> Command Line Interface <<<

This code is based on the pooler cpuminer 2.3.2 release and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use
                          heavy       use to mine Heavycoin
                          mjollnir    use to mine Mjollnircoin
                          fugue256    use to mine Fuguecoin
                          groestl     use to mine Groestlcoin
                          myr-gr      use to mine Myriad-Groestl
                          jackpot     use to mine Jackpotcoin
                          quark       use to mine Quarkcoin
                          anime       use to mine Animecoin
                          nist5       use to mine TalkCoin
                          x11         use to mine DarkCoin

  -d, --devices         gives a comma separated list of CUDA device IDs
                        to operate on. Device IDs start counting from 0!
                        Alternatively give string names of your card like
                        gtx780ti or gt640#2 (matching 2nd gt640 in the PC).

  -f, --diff            Divide difficulty by this factor (std is 1) \n\
  -v, --vote            Heavycoin block vote (default: 512)
  -o, --url=URL         URL of mining server (default: " DEF_RPC_URL ")
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
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

ccminer.exe -t 1 -a heavy -o stratum+tcp://stratum01.heavycoinpool.com:5333 -u <<username.worker>> -p <<workerpassword>> -v 8



Example for Heavycoin Mining on hvc.1gh.com with a dual gpu in your system

ccminer.exe -t 2 -a heavy -o stratum+tcp://hvcpool.1gh.com:5333 -u <<WALLET>> -p x -v 8



Example for Fuguecoin solo-mining with 4 gpu's in your system and a Fuguecoin-wallet running on localhost

ccminer.exe -q -s 1 -t 4 -a fugue256 -o http://localhost:9089 -u <<myusername>> -p <<mypassword>>


Example for Fuguecoin pool mining on dwarfpool.com with all your GPUs

ccminer.exe -q -a fugue256 -o stratum+tcp://erebor.dwarfpool.com:3340 -u YOURWALLETADDRESS.1 -p YOUREMAILADDRESS


Example for Groestlcoin solo mining

ccminer.exe -q -s 1 -a groestl -o http://127.0.0.1:1441 -u USERNAME -p PASSWORD


For solo-mining you typically use -o 127.0.0.1:xxxx where xxxx represents
the RPC portnumber specified in your wallet's .conf file and you have to
pass the same username and password with -O as specified in the wallet's
.conf file. The wallet must also be started with the -server option and
the server flag in the wallet's .conf file set to 1


>>> Additional Notes <<<

This code should be running on nVidia GPUs ranging from compute capability
3.0 up to compute capability 3.5. Support for Compute 2.0 has been dropped
so we can more efficiently implement new algorithms using the latest hardware
features.

>>> RELEASE HISTORY <<<

  June 15th 2014  add X13 and Diamond Groestl support.
                  Thanks to tsiv and to Bombadil for the contributions!

  June 14th 2014  released Killer Groestl quad version which I deem
                  sufficiently hard to port over to AMD. It isn't
                  the fastest option for Compute 3.5 and 5.0 cards,
                  but it is still much faster than the table based
                  versions.

  May 10th 2014   added X11, but without the bells & whistles
                  (no killer Groestl, SIMD hash quite slow still)

  May 6th 2014    this adds the quark and animecoin algorithms.

  May 3rd 2014    add the MjollnirCoin hash algorithm for the upcomin
                  MjollnirCoin relaunch.

                  Add the -f (--diff) option to adjust the difficulty
                  e.g. for the erebor Dwarfpool myr-gr SaffronCoin pool.
                  Use -f 256 there.

  May 1st 2014    adapt the Jackpot algorithms to changes made by the
                  coin developers. We keep our unique nVidia advantage
                  because we have a way to break up the divergence.
                  NOTE: Jackpot Hash now requires Compute 3.0 or later.

  April, 27 2014  this release adds Myriad-Groestl and Jackpot Coin.
                  we apply an optimization to Jackpot that turns this
                  into a Keccak-only CUDA coin ;) Jackpot is tested with
                  solo--mining only at the moment.

  March, 27 2014  Heavycoin exchange rates soar, and as a result this coin
                  gets some love: We greatly optimized the Hefty1 kernel
                  for speed. Expect some hefty gains, especially on 750Ti's!

                  By popular demand, we added the -d option as known from
                  cudaminer.

                  different compute capability builds are now provided until
                  we figure out how to pack everything into a single executable
                  in a Windows build.

  March, 24 2014  fixed Groestl pool support

                  went back to Compute 1.x for cuda_hefty1.cu kernel by
                  default after numerous reports of ccminer v0.2/v0.3
                  not working with HeavyCoin for some people.

  March, 23 2014  added Groestlcoin support. stratum status unknown
                  (the only pool is currently down for fixing issues)

  March, 21 2014  use of shared memory in Fugue256 kernel boosts hash rates
                  on Fermi and Maxwell devices. Kepler may suffer slightly
                  (3-5%)

                  Fixed Stratum for Fuguecoin. Tested on dwarfpool.

  March, 18 2014  initial release.


>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner, Christian H. (Germany): CUDA implementation 

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler), it's original HVC-fork
and the HVC-fork available at hvc.1gh.com

Source code is included to satisfy GNU GPL V3 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
   Christian H. ( Chris84 )
