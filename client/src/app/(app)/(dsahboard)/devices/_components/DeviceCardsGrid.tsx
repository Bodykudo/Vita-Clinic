'use client';

import { useEffect } from 'react';
import axios from 'axios';
import { useQuery } from '@tanstack/react-query';

import DeviceCard from './DeviceCard';
import DeviceCardSkeleton from './DeviceCardSkeleton';
import FiltersBar from '@/components/FiltersBar';
import Pagination from '@/components/Pagination';

import useAccessToken from '@/hooks/useAccessToken';
import useUserRole from '@/hooks/useUserRole';
import { useFiltersStore } from '@/hooks/useFiltersStore';

import type { Device } from '@/types/devices.type';

export default function DeviceCardsGrid() {
  const accessToken = useAccessToken();
  const { role } = useUserRole();

  const {
    sortBy,
    setSortBy,
    searchValue,
    currentPage,
    countPerPage,
    currentStatus,
    reset,
  } = useFiltersStore();

  const {
    data: devices,
    refetch,
    isLoading,
  } = useQuery({
    queryKey: [
      `devices_page_${currentPage}_count_${countPerPage}_status_${currentStatus}_sort_${sortBy}_search_${searchValue}`,
    ],
    queryFn: async () => {
      const response = await axios.get(
        `${process.env.NEXT_PUBLIC_API_URL}/devices?page=${currentPage}&limit=${countPerPage}&status=${currentStatus}&value=${searchValue}&sort=${sortBy}`,
        {
          headers: {
            authorization: `Bearer ${accessToken}`,
          },
        }
      );

      return response.data as Device[];
    },
    enabled: !!accessToken,
  });

  useEffect(() => {
    reset();
    setSortBy('purchaseDate-desc');
  }, []);

  return (
    <>
      <FiltersBar
        refetch={refetch}
        statusFilter
        searchFilter
        searchPlaceholder="Search by device name or manufacturer"
        sortingEnabled
        sortByNameEnabled
        sortByPurchaseDateEnabled
        addNewButton={role === 'admin'}
        addNewRoute="/devices/new"
        addNewContent="New Device"
      />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        {isLoading &&
          Array.from({ length: 7 }).map((_, index) => (
            <DeviceCardSkeleton key={index} />
          ))}
        {devices && devices.length === 0 && <p>No devices found</p>}
        {devices &&
          devices.length > 0 &&
          devices.map((device) => (
            <DeviceCard
              id={device.id}
              key={device.id}
              deviceName={device.name}
              deviceImage={
                device.imageURL ||
                'https://www.dicardiology.com/sites/default/files/field/image/CT%20scan%20with%20patient%20and%20tech_from%20RSNA%202012%20PR%20Toshiba_Aquilion%20ONE%20ViSION%204.jpg'
              }
              manufacturer={device.manufacturer.name}
              lastMaintenanceDate={'2023-02-15'}
              purchaseDate={device.purchaseDate}
              status={device.status}
              serialNumber={device.serialNumber}
            />
          ))}
      </div>

      <Pagination
        previousDisabled={currentPage === 1 || isLoading}
        nextDisabled={(devices && devices.length < countPerPage) || isLoading}
      />
    </>
  );
}
