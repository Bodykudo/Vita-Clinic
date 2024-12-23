'use client';

import Link from 'next/link';
import { useSession } from 'next-auth/react';
import { format, parseISO } from 'date-fns';
import {
  ArrowRight,
  Check,
  CheckCheck,
  CircleOff,
  Timer,
  X,
} from 'lucide-react';

import { Card } from '@/components/ui/card';
import AppointmentDropdownMenu from './AppointmentDropdownMenu';
import { Button, buttonVariants } from '@/components/ui/button';

import useUserRole from '@/hooks/useUserRole';
import { cn } from '@/lib/utils';

import type { AppointmentStatus } from '@/types/appointments.type';
import type { Insurance } from '@/types/emr.type';

interface AppointmentsListItemProps {
  id: string;
  appointmentNumber: number;
  patientId: string;
  patientName: string;
  doctorId: string;
  doctorName: string;
  status: AppointmentStatus;
  bookedAt: string;
  appointmentDate: string;
  cancelledAt: string;
  insurance?: Insurance;
  queryKey?: string;
}

const appointmentStatus = {
  cancelled: {
    icon: X,
    textColor: 'text-red-800',
    backgroundColor: 'bg-red-800/10',
  },
  completed: {
    icon: CheckCheck,
    textColor: 'text-green-700',
    backgroundColor: 'bg-green-700/10',
  },
  pending: {
    icon: Timer,
    textColor: 'text-orange-700',
    backgroundColor: 'bg-orange-700/10',
  },
  approved: {
    icon: Check,
    textColor: 'text-blue-700',
    backgroundColor: 'bg-blue-700/10',
  },
  rejected: {
    icon: CircleOff,
    textColor: 'text-yellow-700',
    backgroundColor: 'bg-yellow-700/10',
  },
};

export default function AppointmentsListItem({
  id,
  appointmentNumber,
  patientId,
  patientName,
  doctorId,
  doctorName,
  status,
  bookedAt,
  appointmentDate,
  cancelledAt,
  insurance,
  queryKey,
}: AppointmentsListItemProps) {
  const session = useSession();
  const { role } = useUserRole();
  const currentStatus = appointmentStatus[status];

  return (
    <Card
      key={id}
      className="flex items-center justify-between border-black/5 p-4 dark:border-gray-800"
    >
      <div className="flex items-center gap-x-4">
        <div
          className={cn('w-fit rounded-md p-2', currentStatus.backgroundColor)}
        >
          <currentStatus.icon
            className={cn('h-8 w-8', currentStatus.textColor)}
          />
        </div>
        <div className="flex flex-col">
          <p className="font-medium">
            Appointment by{' '}
            <Link
              href={`/profile/${patientId}`}
              className="text-primary transition-all hover:text-primary/80"
            >
              {patientName}
            </Link>
          </p>
          <p className="text-sm text-muted-foreground">
            Booked at {format(parseISO(bookedAt), 'MMM dd, yyyy')}
          </p>
          {status === 'pending' ||
            (status === 'approved' && (
              <p className="text-sm text-muted-foreground">
                Appointment to Dr.{' '}
                <Link
                  href={`/profile/${doctorId}`}
                  className="text-primary transition-all hover:text-primary/80"
                >
                  {doctorName}
                </Link>{' '}
                on {format(parseISO(appointmentDate), 'MMM dd, yyyy - hh:mm a')}
              </p>
            ))}
          {status === 'cancelled' && (
            <p className="text-sm text-muted-foreground">
              Cancelled on {format(parseISO(cancelledAt), 'MMM dd, yyyy')},
              booked by {format(parseISO(appointmentDate), 'MMM dd, yyyy')}
            </p>
          )}
          {status === 'completed' && (
            <p className="text-sm text-muted-foreground">
              Appointment to Dr.{' '}
              <Link
                href={`/profile/${doctorId}`}
                className="text-primary transition-all hover:text-primary/80"
              >
                {doctorName}
              </Link>{' '}
              on {format(parseISO(appointmentDate), 'MMM dd, yyyy')}
            </p>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {(role === 'admin' || patientId === session.data?.user.id) && (
          <AppointmentDropdownMenu
            id={id}
            appointmentNumber={appointmentNumber}
            status={status}
            hasInsurance={
              insurance
                ? new Date(insurance.policyEndDate) >= new Date()
                : false
            }
            queryKey={queryKey}
          />
        )}

        <Link
          href={`/appointments/${id}`}
          className={buttonVariants({
            variant: 'ghost',
          })}
        >
          <ArrowRight className="h-5 w-5" />
        </Link>
      </div>
    </Card>
  );
}
