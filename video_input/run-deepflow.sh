if [ "$#" -ne 3 ]; then
  echo "This is an auxiliary script for makeOptFlow.sh. No need to call this script directly."
  exit 1
fi
if [ ! -f deepmatching-static ] && [ ! -f deepflow2-static ]; then
  echo "Place deepflow2-static and deepmatching-static in this directory."
  exit 1
fi

./deepmatching-static $1 $2 -nt 0 | ./deepflow2-static $1 $2 $3 -match