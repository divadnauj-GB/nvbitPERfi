#!/bin/bash
set -e

VAR_LOG_DIR=${1}
HOME_DIR=${2}
SUDOERS_ADD=${3}

RC_LOCAL_PATH=/etc/rc.local

rad_path="${VAR_LOG_DIR}"/radiation-benchmarks
echo "-- Creating ${rad_path} dir"
set -x
sudo mkdir -p "${rad_path}"/log

chmod ugo+w "${rad_path}"
chmod ugo+w "${rad_path}"/log

# Enable the telnet run reboot
if grep -Fq "${SUDOERS_ADD}" /etc/sudoers
then
    # code if found
    echo "+ No need to enable reboot for this user, already enabled"
else
    # code if not found
    echo "+ Enabling reboot for this user"
    echo "${SUDOERS_ADD}" >> /etc/sudoers
fi

chmod 777 "${RC_LOCAL_PATH}"
chmod 777 "${HOME_DIR}"/atBoot.sh


