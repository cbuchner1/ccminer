<?php
/* ccminer API sample UI */

$host = 'http://localhost/api/'; // 'http://'.$_SERVER['SERVER_NAME'].'/api/';
$configs = array(
	'LOCAL'=>'local-sample.php',
	//'EPSYTOUR'=>'epsytour.php', /* copy local.php file and edit target IP:PORT */
);

function getdataFromPears()
{
	global $host, $configs;
	$data = array();
	foreach ($configs as $name => $conf) {

		$json = file_get_contents($host.$conf);

		$data[$name] = json_decode($json, TRUE);
	}
	return $data;
}

function translateField($key)
{
	$intl = array();
	$intl['NAME'] = 'Software';
	$intl['VER'] = 'Version';

	$intl['ALGO'] = 'Algorithm';
	$intl['KHS'] = 'Hashrate (kH/s)';
	$intl['ACC'] = 'Accepted shares';
	$intl['ACCMN'] = 'Accepted / mn';
	$intl['REJ'] = 'Rejected';
	$intl['UPTIME'] = 'Miner uptime';

	$intl['TEMP'] = 'T°c';
	$intl['FAN'] = 'Fan %';

	if (isset($intl[$key]))
		return $intl[$key];
	else
		return $key;
}

function displayData($data)
{
	$htm = '';
	foreach ($data as $name => $stats) {
		$htm .= '<table id="tb_'.$name.'" class="stats">'."\n";
		$htm .= '<tr><th class="machine" colspan="2">'.$name."</th></tr>\n";
		foreach ($stats['summary'] as $key=>$val) {
			if (!empty($val))
				$htm .= '<tr><td class="key">'.translateField($key).'</td>'.
					'<td class="val">'.$val."</td></tr>\n";
		}
		foreach ($stats['stats'] as $g=>$gpu) {
			$htm .= '<tr><th class="gpu" colspan="2">'.$g."</th></tr>\n";
			foreach ($gpu as $key=>$val) {
				if (!empty($val))
				$htm .= '<tr><td class="key">'.translateField($key).'</td>'.
					'<td class="val">'.$val."</td></tr>\n";
			}
		}
		$htm .= "</table>\n";
	}
	return $htm;
}

$data = getdataFromPears();

?>
<html>
<head>
<title>ccminer rig api sample</title>
<style type="text/css">
body { color:#cccccc; background:#1d1d1d; margin:30px 30px 0px 30px; padding:0px; font-size:.8em; font-family:Arial,Helvetica,sans-serif; }
a { color:#aaaaaa; text-decoration: none; }
a:focus { outline-style:none; }
.clear { clear: both; }
div#page, div#header, div#footer {
	margin: auto;
	width: 950px;
	box-shadow: 0 5px 10px rgba(0, 0, 0, 0.15);
}
div#page {
	padding-top: 8px;
	background: #252525;
	min-height: 820px;
}
div#header {
	background: rgba(65, 65, 65, 0.85);
	height: 50px;
	margin-bottom: 24px;
	padding-left: 8px;
}
div#footer {
	background: rgba(25, 25, 25, 0.85);
	height: 0px;
	margin-bottom: 40px;
	text-align: center;
	color: #666666;
	text-shadow: rgba(0, 0, 0, 0.8) 0px 1px 0px;
	padding-top: 8px;
}
#header h1 { padding: 12px; font-size: 20px; }
#footer p { margin: 12px 24px; }

table.stats { width: 280px; margin: 4px 16px; display: inline-block; }
th.machine { color: darkcyan; padding: 16px 0px 0px 0px; text-align: left; border-bottom: 1px solid gray; }
th.gpu { color: white; padding: 3px 3px; font: bolder; text-align: left; background: rgba(65, 65, 65, 0.85); }
td.key { width: 40px; max-width: 120px; }
td.val { width: 70px; max-width: 180px; color: white; }

</style>
</head>
<body>
<div id="header">
<h1>ccminer monitoring API RIG sample</h1>
</div>

<div id="page">
<?=displayData($data)?>
</div>

<div id="footer">
<p>&copy; 2014 <a href="http://github.com/tpruvot/ccminer">tpruvot@github</a></p>
</div>

</body>
</html>
