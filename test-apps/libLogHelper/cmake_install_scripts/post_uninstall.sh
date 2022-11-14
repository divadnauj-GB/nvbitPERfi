#!/bin/bash
set -x
SUDOERS_STR=${1}
echo "+ Removing sudoers reboot enabled"
sed -i "s%$SUDOERS_STR%%g" /etc/sudoers

