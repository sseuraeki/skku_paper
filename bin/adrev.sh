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
dates=$date
sdate=$date
edate=$(gdate +%Y-%m-%d -d "$date +90 day")

while [ "$date" != "$edate" ]; do
    date=$(gdate +%Y-%m-%d -d "$date +1 day")
    dates="$dates $date"
done
# ------

# adclick data ------
mkdir -p $DATADIR/click

adclick=""
for date in $dates; do
    logdate=$(echo $date | sed s/-//g)
    filename=$DATADIR/click/click.$date.csv
    if [ -s "$filename" ]; then
        echo_info "$filename exists, skipping ..."
        adclick="$adclick $filename"
    else
        echo_info "[ $filename ]"
        query="
        SELECT
            appid,
            logdate,
            pid,
            lower(geo) AS geo,
            c AS campaign,
            count(*) AS click
        FROM appsflyer.logdata
        WHERE
            logdate = '$logdate' AND
            event = 'adclick' AND
            TO_DATE(install_time) = '$sdate' AND
            geo = '$af_geo' AND
            pid != 'Organic'
        GROUP BY
            appid, logdate, pid, geo, c"
        python $SCRIPTPATH/bin/query_impala.py "$query" | sed 's/,uk,/,gb,/' | awk -F, -v OFS=, '{if($3=="ironsource_int" || $3=="applovin_int") {$5=$5"@"$4} print}' > $filename
        adclick="$adclick $filename"
    fi
done
awk 'FNR>1 || NR==1' $adclick > $DATADIR/adclick.csv
# ------

# cpc data ------
mkdir -p $SCRIPTPATH/data/cpc

token=$(get_roimon_token)

cpc=""
for date in $dates; do
    filename=$SCRIPTPATH/data/cpc/cpc.$date.csv
    if [ -s "$filename" ]; then
        echo_info "$filename exists, skipping ..."
        cpc="$cpc $filename"
    else
        echo_info "[ $filename ]"
        url="http://roimon.datawave.co.kr/api/v3/data?from=$date&to=$date&appid=&geo=$geo&pid=&grouping=date,appid,geo&kpi=adnetpub.all.cpc.value&accessToken=$token"
        curl -s -X GET "$url" | json_to_csv > $filename
        cpc="$cpc $filename"
    fi
done
awk 'FNR>1 || NR==1' $cpc > $DATADIR/cpc.csv
# ------

# calculate ad revenue ------
python $SCRIPTPATH/bin/adrev.py $DATADIR/adclick.csv $DATADIR/cpc.csv
# ------





