#!/usr/bin/perl

# sample script to query ccminer API

my $command = "summary|";

use Socket;
use IO::Socket::INET;

my $sock = new IO::Socket::INET (
    PeerAddr => '127.0.0.1',
    PeerPort => 4068,
    Proto => 'tcp',
    ReuseAddr => 1,
    Timeout => 10,
);

if ($sock) {

    print $sock $command;
    my $res = "";

    while(<$sock>) {
        $res .= $_;
    }

    close($sock);
    print("$res\n");

} else {

    print("ccminer socket failed\n");

}
