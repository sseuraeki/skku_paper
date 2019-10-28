#!/bin/bash
. $masterAPI/bin/env.sh

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
[[ $mode =~ ^(y|n)$ ]] || echo_exit "$0 $* >>>> -m y|n (y: labels)";
export SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export DATADIR=$SCRIPTPATH/data/$date.$geo

if [ "$mode" == "y" ]; then
	export JOINDIR=$SCRIPTPATH/joined/labeled/$date.$geo
else
	export JOINDIR=$SCRIPTPATH/joined/unlabeled/$date.$geo
fi

[[ -d "$DATADIR" ]] || echo_exit "ERROR: $DATADIR does not exist"
mkdir -p $JOINDIR
# ------

# base ------
filename1=$JOINDIR/base.csv
if [ -s "$filename1" ]; then
	echo_info "$filename1 exists, skipping ..."
else
	echo_info "[ $filename1 ]"
	python $SCRIPTPATH/bin/tr_base.py $DATADIR/base.csv > $filename1
fi
# ------

# retention ------
filename2=$JOINDIR/retention.csv
if [ -s "$filename2" ]; then
	echo_info "$filename2 exists, skipping ..."
else
	echo_info "[ $filename2 ]"
	python $SCRIPTPATH/bin/tr_roimon.py $DATADIR/retention.csv > $filename2
fi
# ------

# return ------
filename3=$JOINDIR/return.csv
if [ -s "$filename3" ]; then
	echo_info "$filename3 exists, skipping ..."
else
	echo_info "[ $filename3 ]"
	python $SCRIPTPATH/bin/tr_roimon.py $DATADIR/return.csv > $filename3
fi
# ------

# activity metrics ------
filename4=$JOINDIR/activity.csv
if [ -s "$filename4" ]; then
	echo_info "$filename4 exists, skipping ..."
else
	echo_info "[ $filename4 ]"
	python $SCRIPTPATH/bin/tr_activity1.py $DATADIR/activity.split.csv > $filename4
fi
# ------

# activity metrics 2 ------
filename5=$JOINDIR/activity2.csv
if [ -s "$filename5" ]; then
	echo_info "$filename5 exists, skipping ..."
else
	echo_info "[ $filename5 ]"
	python $SCRIPTPATH/bin/tr_activity2.py $DATADIR/activity.cpc.csv > $filename5
fi
# ------

if [ "$mode" == "y" ]; then
	# iap & adrev ------
	filename6=$DATADIR/iap.csv
	filename7=$DATADIR/adrev.csv
	# ------

	# join ------
	filename8=$JOINDIR/joined.$date.$geo.csv
	if [ -s "$filename8" ]; then
		echo_info "$filename8 exists, skipping ..."
	else
		echo_info "[ $filename8 ]"
		python $SCRIPTPATH/bin/tr_labeled.py $filename1 $filename2 $filename3 $filename4 $filename5 $filename6 $filename7 > $filename8
	fi
	# ------
fi

if [ "$mode" == "n" ]; then
	filename6=$JOINDIR/joined.$date.$geo.csv
	if [ -s "$filename6" ]; then
		echo_info "$filename6 exists, skipping ..."
	else
		echo_info "[ $filename6 ]"
		python $SCRIPTPATH/bin/tr_unlabeled.py $filename1 $filename2 $filename3 $filename4 $filename5 > $filename6
	fi
fi

