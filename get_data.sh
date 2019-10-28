#!/bin/bash
. $masterAPI/bin/env.sh

# functions ------
function get_roimon() {
	filename=$1
	url=$2

	if [ -s "$filename" ]; then
		echo_info "$filename exists, skipping ..."
	else
		echo_info "[ $filename ]"
		curl -s -X GET "$url" | json_to_csv > $filename
	fi
}

function get_af() {
	filename=$1
	url=$2

	if [ -s "$filename" ]; then
		echo_info "$filename exists, skipping ..."
	else
		echo_info "[ $filename ]"
		curl -s -L "$url"
	fi
}
# ------

# assertions ------
if [ $# -eq 0 ] ; then
    echo_exit "Usage : $0 -d date -g geo -m y/n(y: labels)"
fi

date=""
geo=""
mode=""
while getopts d:g:m: opt ; do
    case "${opt}" in
        d) date=${OPTARG};;
		g) geo=${OPTARG};;
		m) mode=${OPTARG};;
    esac
done

[[ $date =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || echo_exit "$0 $* >>>> -d YYYY-MM-DD";
[[ $geo =~ ^(us|gb|ca|de|au|fr|jp)$ ]] || echo_exit "$0 $* >>>> -g us|gb|ca|de|au|fr|jp";
[[ $mode =~ ^(y|n)$ ]] || echo_exit "$0 $* >>>> -m y|n (y: d365)";
export SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export DATADIR=$SCRIPTPATH/data/$date.$geo
mkdir -p $DATADIR

token=$(get_roimon_token)
if [ "$geo" == "gb" ]; then
	af_geo="UK"
else
	af_geo=$(echo $geo | tr '[:lower:]' '[:upper:]')
fi

TOKEN='dbc2a391-dc14-438f-9b40-1481bb5bad0b'  # appsflyer token
# ------

# base(date, appid, geo, campaign, spend, install) ------
filename=$DATADIR/base.csv
url="http://roimon.datawave.co.kr/api/v3/data?from=$date&to=$date&appid=&geo=$geo&pid=&grouping=date,appid,c1&kpi=campaign.all.spend.value,campaign.all.install.value&accessToken=$token"
get_roimon "$filename" "$url"
# ------

# x set(retention, return, sessions) ------

# retention
kpi="adnetads.all.retention.d1"
for i in 1 2 3 7; do
	kpi="$kpi,adnetads.all.retention.d$i"
done

filename=$DATADIR/retention.csv
url="http://roimon.datawave.co.kr/api/v3/data?from=$date&to=$date&appid=&geo=$geo&pid=&grouping=date,appid&kpi=$kpi&accessToken=$token"
get_roimon "$filename" "$url"

# return
kpi="adnetads.all.return.d1"
for i in 1 2 3 7; do
	kpi="$kpi,adnetads.all.return.d$i"
done

filename=$DATADIR/return.csv
url="http://roimon.datawave.co.kr/api/v3/data?from=$date&to=$date&appid=&geo=$geo&pid=&grouping=date,c1&kpi=$kpi&accessToken=$token"
get_roimon "$filename" "$url"

# activity metrics(appsflyer - dau, mau, arpdau, sessions) ------
kpi="activity_average_dau,activity_average_mau,activity_average_arpdau,activity_sessions"
sdate=$date
edate=$(gdate +%Y-%m-%d -d "$sdate +6 day")
filename=$DATADIR/activity.split.csv
url="https://hq.appsflyer.com/export/master_report/v4?api_token=$TOKEN&app_id=all&geo=$af_geo&from=$sdate&to=$edate&groupings=app_id,install_time&kpis=$kpi&currency=preferred&timezone=preferred"

get_af "$filename" "$url" > $filename
# ------

# activity metrics(roimon - cpi, cpc) ------
filename=$DATADIR/activity.cpc.csv
kpi="adnetads.all.cpi.value,adnetpub.all.cpc.value"
url="http://roimon.datawave.co.kr/api/v3/data?from=$sdate&to=$edate&appid=&geo=$geo&pid=&grouping=date,appid&kpi=$kpi&accessToken=$token"
get_roimon "$filename" "$url" > $filename
# ------

# y set(d365 return) ------

if [ "$mode" == "n" ]; then
	exit
fi

# iap 
filename=$DATADIR/iap.csv
if [ -s "$filename" ]; then
	echo_info "$filename exists, skipping ..."
else
	echo_info "[ $filename ]"
	sh $SCRIPTPATH/bin/iap.sh -d $date -g $geo > $filename
fi

# ad revenue ------
filename=$DATADIR/adrev.csv
echo_info "[ $filename ]"
sh $SCRIPTPATH/bin/adrev.sh -d $date -g $geo > $filename
# ------

echo_info "Finished downloading data"



