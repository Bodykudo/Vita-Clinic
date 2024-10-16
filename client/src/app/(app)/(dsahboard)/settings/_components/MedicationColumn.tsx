'use client';

import { useState } from 'react';
import type { ColumnDef, Row } from '@tanstack/react-table';
import axios, { type AxiosError } from 'axios';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { parseISO, format, formatDistanceToNow } from 'date-fns';
import { ArrowUpDown, MoreHorizontal, Pencil, Trash } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import DeleteAlert from '@/components/DeleteAlert';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useToast } from '@/components/ui/use-toast';

import useAccessToken from '@/hooks/useAccessToken';
import useUserRole from '@/hooks/useUserRole';
import useSettingsStore from '@/hooks/useSettingsStore';

import { capitalize } from '@/lib/utils';
import type { Medication } from '@/types/settings.type';

const ActionsCell = ({ row }: { row: Row<Medication> }) => {
  const accessToken = useAccessToken();
  const { isSuperAdmin } = useUserRole();
  const { openForm, setCurrentMedication } = useSettingsStore();

  const [isDeleting, setIsDeleting] = useState(false);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { mutate: deleteMedication, isPending } = useMutation({
    mutationFn: async () => {
      return await axios.delete(
        `${process.env.NEXT_PUBLIC_API_URL}/settings/medications/${row.original.id}`,
        {
          headers: {
            authorization: `Bearer ${accessToken}`,
          },
        }
      );
    },
    onError: (error) => {
      const axiosError = error as AxiosError<any, any>;

      return toast({
        title: 'Failed to delete medication',
        description:
          axiosError?.response?.data.message ||
          'Medication could not be deleted, please try again.',
        variant: 'destructive',
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['medications'],
      });
      setIsDeleting(false);

      return toast({
        title: 'Medication deleted successfully',
        description: 'Medication has been deleted successfully.',
      });
    },
  });

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="h-8 w-8 p-0">
            <span className="sr-only">Open menu</span>
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Actions</DropdownMenuLabel>
          <DropdownMenuItem
            onClick={() => {
              setCurrentMedication(row.original);
              openForm();
            }}
          >
            <Pencil className="mr-2 h-4 w-4" /> Edit
          </DropdownMenuItem>
          {isSuperAdmin && (
            <DropdownMenuItem onClick={() => setIsDeleting(true)}>
              <Trash className="mr-2 h-4 w-4" /> Delete
            </DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <DeleteAlert
        title={`Delete ${row.original.name}`}
        description={`Are you sure you want to delete ${row.original.name}?`}
        isOpen={isDeleting}
        onClose={() => setIsDeleting(false)}
        onDelete={deleteMedication}
        disabled={isPending}
      />
    </>
  );
};

export const columns: ColumnDef<Medication>[] = [
  {
    id: 'name',
    accessorKey: 'name',
    header: ({ column }) => {
      return (
        <Button
          variant="ghost"
          className="p-0"
          onClick={() => {
            column.toggleSorting(column.getIsSorted() === 'asc');
          }}
        >
          Name
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      );
    },

    cell: ({ row }) => {
      const name = row.original.name;
      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              onClick={() => {
                navigator.clipboard.writeText(name);
              }}
            >
              <span className="font-medium">{name}</span>
            </TooltipTrigger>
            <TooltipContent>Copy to clipboard</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    },
  },
  {
    id: 'dosage',
    header: 'Dosage',
    cell: ({ row }) => {
      const dosageForm = row.original.dosageForm;
      const routeOfAdministration = row.original.routeOfAdministration;

      return (
        <div className="flex flex-col whitespace-nowrap">
          <span>{capitalize(dosageForm)}</span>
          <span className="text-sm text-muted-foreground">
            {capitalize(routeOfAdministration)}
          </span>
        </div>
      );
    },
  },
  {
    id: 'strength',
    header: 'Strength',
    cell: ({ row }) => {
      const strength = row.original.strength;
      const unit = row.original.unit;

      return (
        <div className="flex flex-col whitespace-nowrap">
          <span>
            {strength} {unit}
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: 'description',
    header: 'Description',
    cell: ({ row }) => {
      const description: string | undefined = row.getValue('description');

      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger>
              <p className="max-w-2xl truncate text-sm text-muted-foreground">
                {description || 'No description'}
              </p>
            </TooltipTrigger>

            {description && (
              <TooltipContent side="bottom" className="max-w-xl text-wrap">
                {description}
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      );
    },
  },
  {
    accessorKey: 'createdAt',
    id: 'createdAt',

    header: ({ column }) => {
      return (
        <Button
          variant="ghost"
          className="p-0"
          onClick={() => {
            column.toggleSorting(column.getIsSorted() === 'asc');
          }}
        >
          Created at
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      );
    },
    cell: ({ row }) => {
      const createdAt: string = row.getValue('createdAt');

      return (
        <div className="flex flex-col whitespace-nowrap">
          <span>{format(parseISO(createdAt), 'MMM dd, yyyy')}</span>
          <span className="text-sm text-muted-foreground">
            {formatDistanceToNow(parseISO(createdAt), {
              addSuffix: true,
            })}
          </span>
        </div>
      );
    },
  },
  {
    id: 'actions',
    cell: ({ row }) => <ActionsCell row={row} />,
  },
];
