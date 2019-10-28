#!/bin/bash
. $masterAPI/bin/env.sh

if [ $# -eq 0 ] ; then
    echo_exit "Usage : $0 -d date -g geo"
fi

date=""
geo=""
while getopts d:g: opt ; do
    case "${opt}" in
        d) date=${OPTARG};;
        g) geo=${OPTARG};;
    esac
done

[[ $date =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || echo_exit "$0 $* >>>> -d YYYY-MM-DD";

if [ "$geo" == "gb" ]; then
	af_geo="UK"
else
	af_geo=$(echo $geo | tr '[:lower:]' '[:upper:]')
fi

# get dates ------
sdate=$(echo $date | sed s/-//g)
edate=$(gdate +%Y-%m-%d -d "$date +90 days" | sed s/-//g)
# ------

# query data ------
query="
SELECT
	c AS campaign,
	pid,
	'$geo' AS geo,
	SUM(CAST(v2 AS FLOAT) * CAST(v3 AS INT)) AS iap
FROM appsflyer.logdata
WHERE
	logdate >= '$sdate' AND
	logdate <= '$edate' AND
	event = 'purchase' AND
	geo = '$af_geo' AND
	TO_DATE(install_time) = '$date' AND
	pid != 'Organic'
GROUP BY
	c, pid"

python $SCRIPTPATH/bin/query_impala.py "$query" | awk -F, -v OFS=, '{if($2=="ironsource_int" || $2=="applovin_int") {$1=$1"@"$3} print}'


